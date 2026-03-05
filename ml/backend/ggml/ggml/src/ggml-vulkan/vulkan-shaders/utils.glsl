#ifndef UTILS_COMP
#define UTILS_COMP

// mod and div are expensive and coordinates/dimensions are often power of 2 or equal to 1
uint fastmod(uint a, uint b) {
    if ((b & (b-1)) == 0) {
        return a & (b-1);
    }
    return a % b;
}

uint fastdiv(uint a, uint b) {
    return (a < b) ? 0 : (a / b);
}

void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03, uint ne00, uint ne01, uint ne02, uint ne03) {
    i03 = fastdiv(idx, (ne02*ne01*ne00));
    const uint i03_offset = i03 * ne02*ne01*ne00;
    i02 = fastdiv((idx - i03_offset), (ne01*ne00));
    const uint i02_offset = i02*ne01*ne00;
    i01 = (idx - i03_offset - i02_offset) / ne00;
    i00 = idx - i03_offset - i02_offset - i01*ne00;
}

#endif // UTILS_COMP
