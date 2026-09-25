// Parity check for the IQ2_S MMQ loader's sign application: the packed
// decoder applied the sign as __vsub4(grid ^ vcmpne4(mask), vcmpne4(mask));
// the loader now uses the byte-wise two's-complement identity
// (grid ^ (add*0xff)) + add. Both must produce identical signed bytes for
// every grid magnitude and every 8-bit sign byte.
#include <cstdint>
#include <cstdio>

namespace {

int Vcmpne4(std::uint32_t x) {
  std::uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    r |= (x & (0xFFu << (8 * i))) ? (0xFFu << (8 * i)) : 0u;
  }
  return static_cast<int>(r);
}
std::uint32_t Vsub4(std::uint32_t a, std::uint32_t b) {
  std::uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t av = (a >> (8 * i)) & 0xFFu;
    const std::uint32_t bv = (b >> (8 * i)) & 0xFFu;
    r |= ((av - bv) & 0xFFu) << (8 * i);
  }
  return r;
}
std::uint32_t Xor(std::uint32_t a, std::uint32_t b) {
  std::uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    r |= (((a >> (8 * i)) & 0xFFu) ^ ((b >> (8 * i)) & 0xFFu)) << (8 * i);
  }
  return r;
}
std::uint32_t ApplySignBits(std::uint32_t grid, std::uint32_t add) {
  return (grid ^ (add * 0xffu)) + add;
}

}  // namespace

int main() {
  const std::uint32_t mags[] = {0x08, 0x19, 0x2b};
  bool ok = true;
  for (std::uint32_t mag : mags) {
    const std::uint32_t grid = mag | (mag << 8) | (mag << 16) | (mag << 24);
    for (int sb = 0; sb < 256 && ok; ++sb) {
      // Both nibble halves of the packed sign byte, as the loader splits it.
      const std::uint32_t b0 = (sb & 0x03) | ((sb & 0x0C) >> 2);
      const std::uint32_t b1 = ((sb & 0x30) >> 4) | ((sb & 0xC0) >> 6);
      for (auto bits : {b0, b1}) {
        const std::uint32_t add =
            (bits * 0x00204081u) & 0x01010101u;
        const std::uint32_t fast = ApplySignBits(grid, add);
        // Original: mask the sign bits into byte MSBs, then vsub4.
        const std::uint32_t mask = bits == b0
            ? (static_cast<std::uint32_t>(sb & 0x03) << 7) |
                  (static_cast<std::uint32_t>(sb & 0x0C) << 21)
            : (static_cast<std::uint32_t>(sb & 0x30) << 3) |
                  (static_cast<std::uint32_t>(sb & 0xC0) << 17);
        const std::uint32_t slow =
            Vsub4(Xor(grid, static_cast<std::uint32_t>(Vcmpne4(mask))),
                  static_cast<std::uint32_t>(Vcmpne4(mask)));
        if (fast != slow) {
          ok = false;
          std::printf("MISMATCH mag=%02x sb=%02x bits=%x fast=%08x slow=%08x\n",
                      mag, sb, bits, fast, slow);
          break;
        }
      }
    }
  }
  std::printf(ok ? "IQ2S_SIGN_PARITY_PASS\n" : "IQ2S_SIGN_PARITY_FAIL\n");
  return ok ? 0 : 1;
}
