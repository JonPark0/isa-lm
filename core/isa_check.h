/* Runtime check that the CPU supports the ISA this binary was built for.
 * Compiled without ISA flags so it is safe to run on any CPU of the arch. */
#ifndef ML_ISA_CHECK_H
#define ML_ISA_CHECK_H
#include "core/ml.h"

/* ML_OK, or ML_E_ISA with *missing naming the first absent feature. */
ml_status ml_isa_check(const char **missing);
const char *ml_isa_build_name(void);

#endif
