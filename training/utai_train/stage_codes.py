"""S172: stable CODEs for the training PROGRESS messages.

WHY THIS FILE EXISTS. These strings used to be Chinese literals passed straight to
`reporter.stage(message=...)`, and `TrainingPage.tsx` renders that message as-is when it does
not resolve to a CODE. So every non-Chinese user watched Chinese progress text for the entire
length of every training run. The display side already did the right thing --
`backendErrorMessage(cur.message) ?? cur.message` -- it just never had a CODE to resolve; the
conversion had been left half-done.

Duplicates SHARE a code (no-duplication hard rule): the file list stage and the retrieval index
stage each exist in two lanes, and "loading, about to start" exists in three. The diffusion and
vocoder loading messages stay SEPARATE because they name different things, and someone watching
the bar should be able to tell which one is being loaded.

Two of these carry a count. They use the house `CODE: detail` convention documented at the top
of `src/lib/backendError.ts`: the detail is appended to the localized text in parentheses, so
the number survives without the sentence having to be assembled in python.

WARNING: `reporter.stage(message=...)` is a USER-FACING channel, exactly like a raise. If you
add a stage message, add a CODE here and the three `backend.*` texts -- do not pass a bare
string, in any language.

These are plain top-level literals on purpose: the cross-language CODE gate parses them as the
single source (same contract as `prep_codes.py` and `device.py::AMD_GPU_NOT_FOUND_CODE`).
"""

#: The augmentation pass produced nothing to check.
NO_AUGMENT = 'TRAINING_STAGE_NO_AUGMENT'

#: Writing the training file list and config (rvc + sovits).
FILELIST = 'TRAINING_STAGE_FILELIST'

#: Building the retrieval feature index (rvc + sovits).
BUILD_INDEX = 'TRAINING_STAGE_BUILD_INDEX'

#: Model and data loading, just before the first step (rvc + sovits + sovits_v2).
LOADING = 'TRAINING_STAGE_LOADING'

#: Training the kmeans cluster centroids.
KMEANS = 'TRAINING_STAGE_KMEANS'

#: Preparing the diffusion config and its pretrained model.
DIFF_PREP = 'TRAINING_STAGE_DIFF_PREP'

#: Diffusion model and data loading.
DIFF_LOADING = 'TRAINING_STAGE_DIFF_LOADING'

#: No diffusion pretrain found; the run starts from scratch.
DIFF_NO_PRETRAIN = 'TRAINING_STAGE_DIFF_NO_PRETRAIN'

#: Caching the training data.
CACHING = 'TRAINING_STAGE_CACHING'

#: Vocoder pretrain and data loading.
VOCODER_LOADING = 'TRAINING_STAGE_VOCODER_LOADING'

#: N augmented clips dropped after feature extraction failed. Carries the count.
DROPPED_AUG = 'TRAINING_STAGE_DROPPED_AUG'

#: Splitting slices into the training and validation sets.
SPLIT = 'TRAINING_STAGE_SPLIT'

#: N slices dropped for being shorter than the crop window. Carries the count.
DROPPED_SHORT = 'TRAINING_STAGE_DROPPED_SHORT'
