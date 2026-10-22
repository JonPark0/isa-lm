/* Process entry for every executable. Compiled without ISA flags: nothing
 * built for the target ISA runs until the CPU has been checked. */
#include <stdio.h>
#include "core/isa_check.h"

int ml_main(int argc, char **argv);

int main(int argc, char **argv) {
    const char *missing = NULL;
    if (ml_isa_check(&missing) != ML_OK) {
        fprintf(stderr, "error: CPU lacks %s required by ISA=%s\n", missing, ml_isa_build_name());
        return 3;
    }
    return ml_main(argc, argv);
}
