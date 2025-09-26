void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint row = i / p.ne20;
    const uint col = i - row * p.ne20;

    if (p.mode == 0) {
        // Default
        const uint offset = p.ne00 / 2;
        const uint idx = row * p.ne00 + col;

        data_d[row * offset + col] = D_TYPE(op(float(data_a[idx]), float(data_a[idx + offset])));
    } else if (p.mode == 1) {
        // Swapped
        const uint offset = p.ne00 / 2;
        const uint idx = row * p.ne00 + col;

        data_d[row * offset + col] = D_TYPE(op(float(data_a[idx + offset]), float(data_a[idx])));
    } else {
        // Split
        const uint idx = row * p.ne00 + col;

        data_d[idx] = D_TYPE(op(float(data_a[idx]), float(data_b[idx])));
    }
}
