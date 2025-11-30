
float rope_yarn_ramp(const float low, const float high, const uint i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

uint rope_a_coord(const uint i0, const uint i01, const uint i02, rope_params p) {
#if RMS_NORM_ROPE_FUSION
    // Per-row offset in shared memory
    const uint ix = i0;
#else
    const uint ix = i02*p.nb02 + i01*p.nb01 + i0;
#endif
    return ix;
}

void rope_yarn(const float theta_extrap, const uint i0, out float cos_theta, out float sin_theta, rope_params p) {
    float mscale = p.attn_factor;
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    if (p.ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(p.corr_dims[0], p.corr_dims[1], i0) * p.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    // Backprogagation uses inverted rotation
    if (p.is_back != 0) {
        theta = -theta;
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

void rope_norm(const uint i0, const uint i1, rope_params p) {
    uint ne0 = p.ncols;
    uint ne1 = p.p_delta_rows;

    if (i0 >= ne0) {
        return;
    }

    // i1 is actually i2*nb2+i1, but the rows are contiguous
    const uint i01 = i1 % ne1;
    const uint i02 = i1 / ne1;

    uint idst = i1*ne0 + i0;
    const uint ix = rope_a_coord(i0, i01, i02, p);

    // Fusion optimization: ROPE + VIEW + SET_ROWS..
    // The rope output is viewed as a 1D tensor and offset based on a row index in data_i.
    if (p.set_rows_stride != 0) {
        idst = i01*ne0 + i0;
        idst += rope_data_i[i02].x * p.set_rows_stride;
    }

    if (i0 >= p.n_dims) {
        rope_data_d[idst + 0] = ROPE_D_TYPE(rope_data_a[ix + 0]);
        rope_data_d[idst + 1] = ROPE_D_TYPE(rope_data_a[ix + 1]);

        return;
    }

    const float theta_base = rope_data_pos[i02] * pow(p.theta_scale, i0/2.0f);

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + 1]);

    rope_data_d[idst + 0] = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + 1] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

void rope_neox(const uint i0, const uint i1, rope_params p) {
    uint ne0 = p.ncols;
    uint ne1 = p.p_delta_rows;

    if (i0 >= ne0) {
        return;
    }

    const uint i01 = i1 % ne1;
    const uint i02 = i1 / ne1;

    uint idst = i1*ne0 + i0/2;
    const uint ix = rope_a_coord(i0/2, i01, i02, p);

    // Fusion optimization: ROPE + VIEW + SET_ROWS..
    // The rope output is viewed as a 1D tensor and offset based on a row index in rope_data_i.
    if (p.set_rows_stride != 0) {
        idst = i01*ne0 + i0/2;
        idst += rope_data_i[i02].x * p.set_rows_stride;
    }

    if (i0 >= p.n_dims) {
        rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
        rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);

        return;
    }

    const float theta_base = rope_data_pos[i02] * pow(p.theta_scale, i0/2.0f);

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims/2]);

    rope_data_d[idst + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}


void rope_multi(const uint i0, const uint i1, rope_params p) {
    uint ne0 = p.ncols;
    uint ne1 = p.p_delta_rows;
    uint ne2 = p.ne02;

    if (i0 >= ne0) {
        return;
    }

    const uint i01 = i1 % ne1;
    const uint i02 = i1 / ne1;

    const uint idst = i1*ne0 + i0/2;
    const uint ix = rope_a_coord(i0/2, i01, i02, p);

    if (i0 >= p.n_dims) {
        rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
        rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);

        return;
    }

    const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
    const int sec_w = p.sections[1] + p.sections[0];
    const uint sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (p.is_imrope != 0) {
        if (sector % 3 == 1 && sector < 1 + 3 * p.sections[1]) {
            theta_base = rope_data_pos[i02 + ne2 * 1]*pow(p.theta_scale, i0/2.0f);
        } else if (sector % 3 == 2 && sector < 2 + 3 * p.sections[2]) {
            theta_base = rope_data_pos[i02 + ne2 * 2]*pow(p.theta_scale, i0/2.0f);
        } else if (sector % 3 == 0 && sector < 3 * p.sections[0]) {
            theta_base = rope_data_pos[i02]*pow(p.theta_scale, i0/2.0f);
        //} else {
        //    theta_base = rope_data_pos[i02 + ne2 * 3]*pow(p.theta_scale, i0/2.0f);
        }
    } else {
        if (sector < p.sections[0]) {
            theta_base = rope_data_pos[i02]*pow(p.theta_scale, i0/2.0f);
        }
        else if (sector >= p.sections[0] && sector < sec_w) {
            theta_base = rope_data_pos[i02 + ne2 * 1]*pow(p.theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w && sector < sec_w + p.sections[2]) {
            theta_base = rope_data_pos[i02 + ne2 * 2]*pow(p.theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w + p.sections[2]) {
            theta_base = rope_data_pos[i02 + ne2 * 3]*pow(p.theta_scale, i0/2.0f);
        }
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims/2]);

    rope_data_d[idst + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

void rope_vision(const uint i0, const uint i1, rope_params p) {
    uint ne0 = p.ncols;
    uint ne1 = p.p_delta_rows;
    uint ne2 = p.ne02;

    if (i0 >= ne0) {
        return;
    }

    const uint i01 = i1 % ne1;
    const uint i02 = i1 / ne1;

    const uint idst = i1*ne0 + i0/2;
    const uint ix = rope_a_coord(i0/2, i01, i02, p);

    const int sect_dims = p.sections[0] + p.sections[1];
    const int sec_w = p.sections[1] + p.sections[0];
    const uint sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < p.sections[0]) {
        const uint p0 = sector;
        theta_base = rope_data_pos[i02]*pow(p.theta_scale, p0);
    }
    else if (sector >= p.sections[0] && sector < sec_w) {
        const uint p0 = sector - p.sections[0];
        theta_base = rope_data_pos[i02 + ne2]*pow(p.theta_scale, p0);
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims]);

    rope_data_d[idst + 0]        = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_dims] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

