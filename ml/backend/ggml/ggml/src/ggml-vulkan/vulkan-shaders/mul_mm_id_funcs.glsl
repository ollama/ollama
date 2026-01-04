#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

#ifdef MUL_MAT_ID_USE_SUBGROUPS
shared uvec4 ballots_sh[NUM_WARPS];

void load_row_ids(uint expert_idx, bool nei0_is_pow2, uint ic) {
    _ne1 = 0;
    uint num_elements = p.nei1 * p.nei0;
    uint nei0shift = findLSB(p.nei0);

    uint ids[16];
    uint iter = 0;

    for (uint j = 0; j < num_elements; j += BLOCK_SIZE) {
        // prefetch up to 16 elements
        if (iter == 0) {
            [[unroll]] for (uint k = 0; k < 16; ++k) {
                uint i = j + gl_LocalInvocationIndex + k*BLOCK_SIZE;
                bool in_range = i < num_elements;
                uint ii1;
                if (nei0_is_pow2) {
                    ii1 = i >> nei0shift;
                } else {
                    ii1 = i / p.nei0;
                }
                uint ii0 = i - ii1 * p.nei0;
                ids[k] = in_range ? data_ids[ii1*p.nbi1 + ii0] : 0;
            }
        }
        uint i = j + gl_LocalInvocationIndex;
        bool in_range = i < num_elements;
        uint ii1;
        if (nei0_is_pow2) {
            ii1 = i >> nei0shift;
        } else {
            ii1 = i / p.nei0;
        }
        uint ii0 = i - ii1 * p.nei0;
        uint id = ids[iter++];
        uvec4 ballot = subgroupBallot(in_range && id == expert_idx);

        ballots_sh[gl_SubgroupID] = ballot;
        barrier();

        uint subgroup_base = 0;
        uint total = 0;
        for (uint k = 0; k < gl_NumSubgroups; ++k) {
            if (k == gl_SubgroupID) {
                subgroup_base = total;
            }
            total += subgroupBallotBitCount(ballots_sh[k]);
        }
        barrier();

        uint idx = subgroup_base + subgroupBallotExclusiveBitCount(ballot);
        if (in_range && id == expert_idx && _ne1 + idx >= ic * BN && _ne1 + idx < (ic + 1) * BN) {
            row_ids[_ne1 + idx - ic * BN] = u16vec2(ii0, ii1);
        }
        _ne1 += total;
        iter &= 15;
        if (_ne1 >= (ic + 1) * BN) {
            break;
        }
    }
    barrier();
}
#endif // MUL_MAT_ID_USE_SUBGROUPS
#endif // MUL_MAT_ID
