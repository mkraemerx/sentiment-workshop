# Examination



# Notes
* Special Handling of twitter text
  * Capitalization of Twitter handles uesless for spacy

# Inspiration
* http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
* https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-i-5da6f1c89d84

# Baseline
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 1
counting (train only)..
computing r..
processing files..
iter  1 act 1.539e+03 pre 1.381e+03 delta 2.130e+01 f 2.777e+03 |g| 4.172e+02 CG   7
iter  2 act 1.743e+02 pre 1.503e+02 delta 2.130e+01 f 1.238e+03 |g| 1.043e+02 CG   7
iter  3 act 2.456e+01 pre 2.123e+01 delta 2.130e+01 f 1.064e+03 |g| 3.194e+01 CG   6
iter  4 act 2.861e+00 pre 2.647e+00 delta 2.130e+01 f 1.039e+03 |g| 7.147e+00 CG   8
Accuracy = 74.3513% (745/1002)
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
computing r..
processing files..
iter  1 act 1.879e+03 pre 1.683e+03 delta 1.955e+01 f 2.777e+03 |g| 5.274e+02 CG   8
iter  2 act 1.984e+02 pre 1.680e+02 delta 1.955e+01 f 8.988e+02 |g| 1.328e+02 CG   7
iter  3 act 3.400e+01 pre 2.878e+01 delta 1.955e+01 f 7.004e+02 |g| 4.454e+01 CG   6
iter  4 act 6.098e+00 pre 5.466e+00 delta 1.955e+01 f 6.664e+02 |g| 1.196e+01 CG   7
iter  5 act 4.443e-01 pre 4.150e-01 delta 1.955e+01 f 6.603e+02 |g| 2.613e+00 CG   7
Accuracy = 75.9481% (761/1002)
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 123
counting (train only)..
computing r..
processing files..
iter  1 act 1.977e+03 pre 1.764e+03 delta 1.724e+01 f 2.777e+03 |g| 6.025e+02 CG   8
iter  2 act 2.245e+02 pre 1.892e+02 delta 1.724e+01 f 8.008e+02 |g| 1.500e+02 CG   7
iter  3 act 3.881e+01 pre 3.254e+01 delta 1.724e+01 f 5.763e+02 |g| 5.109e+01 CG   5
iter  4 act 7.781e+00 pre 6.779e+00 delta 1.724e+01 f 5.375e+02 |g| 1.553e+01 CG   6
iter  5 act 9.779e-01 pre 9.088e-01 delta 1.724e+01 f 5.297e+02 |g| 3.655e+00 CG   7
Accuracy = 74.5509% (747/1002)
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 1234
counting (train only)..
computing r..
processing files..
iter  1 act 2.022e+03 pre 1.803e+03 delta 1.615e+01 f 2.777e+03 |g| 6.201e+02 CG   9
iter  2 act 2.357e+02 pre 1.991e+02 delta 1.615e+01 f 7.551e+02 |g| 1.528e+02 CG   9
iter  3 act 4.007e+01 pre 3.357e+01 delta 1.615e+01 f 5.194e+02 |g| 5.200e+01 CG   6
iter  4 act 8.444e+00 pre 7.255e+00 delta 1.615e+01 f 4.793e+02 |g| 1.630e+01 CG   8
iter  5 act 1.077e+00 pre 9.827e-01 delta 1.615e+01 f 4.708e+02 |g| 4.447e+00 CG   7
Accuracy = 70.9581% (711/1002)

## with only 0.1 test data
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
vocab len 88294
computing r..
processing files..
iter  1 act 2.115e+03 pre 1.890e+03 delta 2.010e+01 f 3.125e+03 |g| 6.021e+02 CG   8
iter  2 act 2.350e+02 pre 1.978e+02 delta 2.010e+01 f 1.010e+03 |g| 1.522e+02 CG   7
iter  3 act 4.298e+01 pre 3.623e+01 delta 2.010e+01 f 7.748e+02 |g| 5.201e+01 CG   6
iter  4 act 8.357e+00 pre 7.442e+00 delta 2.010e+01 f 7.318e+02 |g| 1.441e+01 CG   8
iter  5 act 6.707e-01 pre 6.167e-01 delta 2.010e+01 f 7.235e+02 |g| 3.130e+00 CG   8
Accuracy = 75.6487% (379/501)

## with cv=5
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
vocab len 88617
computing r..
processing files..
iter  1 act 1.692e+03 pre 1.515e+03 delta 1.839e+01 f 2.500e+03 |g| 5.041e+02 CG   8
iter  2 act 1.773e+02 pre 1.486e+02 delta 1.839e+01 f 8.080e+02 |g| 1.270e+02 CG   6
iter  3 act 3.411e+01 pre 2.908e+01 delta 1.839e+01 f 6.307e+02 |g| 4.245e+01 CG   6
iter  4 act 5.669e+00 pre 5.083e+00 delta 1.839e+01 f 5.965e+02 |g| 1.222e+01 CG   7
iter  5 act 4.446e-01 pre 4.044e-01 delta 1.839e+01 f 5.909e+02 |g| 2.689e+00 CG   7
iter  1 act 1.692e+03 pre 1.516e+03 delta 1.854e+01 f 2.499e+03 |g| 5.166e+02 CG   8
iter  2 act 1.776e+02 pre 1.501e+02 delta 1.854e+01 f 8.072e+02 |g| 1.299e+02 CG   7
iter  3 act 3.111e+01 pre 2.628e+01 delta 1.854e+01 f 6.296e+02 |g| 4.354e+01 CG   6
iter  4 act 5.550e+00 pre 4.935e+00 delta 1.854e+01 f 5.985e+02 |g| 1.216e+01 CG   7
iter  5 act 4.201e-01 pre 3.887e-01 delta 1.854e+01 f 5.930e+02 |g| 2.589e+00 CG   7
iter  1 act 1.692e+03 pre 1.514e+03 delta 1.822e+01 f 2.500e+03 |g| 5.304e+02 CG   8
iter  2 act 1.836e+02 pre 1.551e+02 delta 1.822e+01 f 8.079e+02 |g| 1.333e+02 CG   7
iter  3 act 3.270e+01 pre 2.756e+01 delta 1.822e+01 f 6.242e+02 |g| 4.488e+01 CG   6
iter  4 act 6.137e+00 pre 5.447e+00 delta 1.822e+01 f 5.915e+02 |g| 1.246e+01 CG   7
iter  5 act 4.863e-01 pre 4.478e-01 delta 1.822e+01 f 5.854e+02 |g| 2.698e+00 CG   7
iter  1 act 1.695e+03 pre 1.520e+03 delta 1.853e+01 f 2.499e+03 |g| 5.175e+02 CG   8
iter  2 act 1.769e+02 pre 1.495e+02 delta 1.853e+01 f 8.042e+02 |g| 1.291e+02 CG   7
iter  3 act 3.054e+01 pre 2.577e+01 delta 1.853e+01 f 6.272e+02 |g| 4.319e+01 CG   5
iter  4 act 5.786e+00 pre 5.180e+00 delta 1.853e+01 f 5.967e+02 |g| 1.171e+01 CG   7
iter  5 act 4.368e-01 pre 4.076e-01 delta 1.853e+01 f 5.909e+02 |g| 2.432e+00 CG   7
iter  1 act 1.694e+03 pre 1.518e+03 delta 1.844e+01 f 2.499e+03 |g| 5.216e+02 CG   8
iter  2 act 1.794e+02 pre 1.516e+02 delta 1.844e+01 f 8.053e+02 |g| 1.304e+02 CG   7
iter  3 act 3.163e+01 pre 2.671e+01 delta 1.844e+01 f 6.260e+02 |g| 4.377e+01 CG   6
iter  4 act 5.727e+00 pre 5.065e+00 delta 1.844e+01 f 5.943e+02 |g| 1.198e+01 CG   7
iter  5 act 4.744e-01 pre 4.389e-01 delta 1.844e+01 f 5.886e+02 |g| 2.563e+00 CG   7
Cross Validation Accuracy = 80.0133%
Accuracy = 69.0619% (346/501)

## with cv=2
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
vocab len 88475
computing r..
processing files..
iter  1 act 1.054e+03 pre 9.521e+02 delta 1.594e+01 f 1.562e+03 |g| 2.890e+02 CG   8
iter  2 act 9.164e+01 pre 7.800e+01 delta 1.594e+01 f 5.083e+02 |g| 6.892e+01 CG   6
iter  3 act 1.430e+01 pre 1.234e+01 delta 1.594e+01 f 4.166e+02 |g| 2.204e+01 CG   6
iter  4 act 1.916e+00 pre 1.719e+00 delta 1.594e+01 f 4.023e+02 |g| 5.731e+00 CG   6
iter  5 act 1.354e-01 pre 1.281e-01 delta 1.594e+01 f 4.004e+02 |g| 1.198e+00 CG   7
iter  1 act 1.051e+03 pre 9.441e+02 delta 1.511e+01 f 1.562e+03 |g| 3.156e+02 CG   7
iter  2 act 1.048e+02 pre 8.954e+01 delta 1.511e+01 f 5.113e+02 |g| 7.722e+01 CG   6
iter  3 act 1.586e+01 pre 1.352e+01 delta 1.511e+01 f 4.065e+02 |g| 2.458e+01 CG   5
iter  4 act 2.509e+00 pre 2.252e+00 delta 1.511e+01 f 3.906e+02 |g| 6.171e+00 CG   6
iter  5 act 1.549e-01 pre 1.492e-01 delta 1.511e+01 f 3.881e+02 |g| 1.207e+00 CG   7
Cross Validation Accuracy = 77.9059%
Accuracy = 71.6567% (359/501)

## without cv and s=2
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 1

counting (train only)..
vocab len 18979
computing r..
processing files..
iter  1 act 3.969e+03 pre 3.903e+03 delta 1.839e+01 f 4.508e+03 |g| 1.788e+03 CG   8
iter  2 act -2.540e+01 pre 2.210e+02 delta 5.139e+00 f 5.391e+02 |g| 2.039e+02 CG  14
cg reaches trust region boundary
iter  2 act 1.426e+02 pre 1.575e+02 delta 2.056e+01 f 5.391e+02 |g| 2.039e+02 CG   6
iter  3 act -8.541e+01 pre 8.274e+01 delta 2.658e+00 f 3.965e+02 |g| 6.226e+01 CG  17
cg reaches trust region boundary
iter  3 act 4.472e+01 pre 4.884e+01 delta 1.063e+01 f 3.965e+02 |g| 6.226e+01 CG   6
iter  4 act -2.422e+02 pre 4.606e+01 delta 1.940e+00 f 3.518e+02 |g| 3.338e+01 CG  18
cg reaches trust region boundary
iter  4 act 1.103e+01 pre 2.101e+01 delta 1.366e+00 f 3.518e+02 |g| 3.338e+01 CG   5
cg reaches trust region boundary
iter  5 act 9.289e+00 pre 1.344e+01 delta 1.053e+00 f 3.407e+02 |g| 3.613e+01 CG   5
cg reaches trust region boundary
iter  6 act 4.719e+00 pre 6.869e+00 delta 8.058e-01 f 3.314e+02 |g| 2.570e+01 CG   6
cg reaches trust region boundary
iter  7 act 2.720e+00 pre 4.352e+00 delta 6.011e-01 f 3.267e+02 |g| 1.859e+01 CG   6
cg reaches trust region boundary
iter  8 act 1.694e+00 pre 2.608e+00 delta 4.609e-01 f 3.240e+02 |g| 1.292e+01 CG   5
cg reaches trust region boundary
iter  9 act 1.001e+00 pre 1.415e+00 delta 3.614e-01 f 3.223e+02 |g| 9.455e+00 CG   5
cg reaches trust region boundary
iter 10 act 6.893e-01 pre 9.567e-01 delta 2.949e-01 f 3.213e+02 |g| 7.794e+00 CG   5
Accuracy = 72.6547% (364/501)

(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
vocab len 88417
computing r..
processing files..
iter  1 act 4.272e+03 pre 4.230e+03 delta 1.220e+01 f 4.508e+03 |g| 2.413e+03 CG   7
iter  2 act -8.064e+02 pre 1.569e+02 delta 1.915e+00 f 2.362e+02 |g| 2.188e+02 CG  20
cg reaches trust region boundary
iter  2 act 8.108e+01 pre 8.357e+01 delta 7.661e+00 f 2.362e+02 |g| 2.188e+02 CG   5
iter  3 act -4.207e+02 pre 6.559e+01 delta 1.573e+00 f 1.551e+02 |g| 9.837e+01 CG  21
cg reaches trust region boundary
iter  3 act 3.495e+01 pre 3.545e+01 delta 6.293e+00 f 1.551e+02 |g| 9.837e+01 CG   5
cg reaches trust region boundary
iter  4 act -6.844e+02 pre 3.596e+01 delta 1.573e+00 f 1.202e+02 |g| 3.860e+01 CG  17
cg reaches trust region boundary
iter  4 act 1.451e+00 pre 1.547e+01 delta 7.866e-01 f 1.202e+02 |g| 3.860e+01 CG   5
cg reaches trust region boundary
iter  5 act 1.472e+01 pre 1.643e+01 delta 3.146e+00 f 1.187e+02 |g| 9.855e+01 CG   6
cg reaches trust region boundary
iter  6 act -1.396e+02 pre 9.888e+00 delta 7.866e-01 f 1.040e+02 |g| 1.734e+01 CG  10
cg reaches trust region boundary
iter  6 act -2.752e+00 pre 3.716e+00 delta 2.811e-01 f 1.040e+02 |g| 1.734e+01 CG   6
cg reaches trust region boundary
iter  6 act 1.406e+00 pre 1.700e+00 delta 1.124e+00 f 1.040e+02 |g| 1.734e+01 CG   4
cg reaches trust region boundary
iter  7 act -1.095e+01 pre 3.333e+00 delta 2.811e-01 f 1.026e+02 |g| 1.091e+01 CG   6
cg reaches trust region boundary
iter  7 act 5.234e-01 pre 1.133e+00 delta 1.916e-01 f 1.026e+02 |g| 1.091e+01 CG   4
cg reaches trust region boundary
iter  8 act 8.918e-01 pre 1.009e+00 delta 7.664e-01 f 1.021e+02 |g| 1.828e+01 CG   4
Accuracy = 74.6507% (374/501)

(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 123
counting (train only)..
vocab len 179123
computing r..
processing files..
iter  1 act 4.326e+03 pre 4.289e+03 delta 1.011e+01 f 4.508e+03 |g| 2.757e+03 CG   8
iter  2 act -4.152e+01 pre 1.230e+02 delta 1.817e+00 f 1.821e+02 |g| 2.427e+02 CG  16
cg reaches trust region boundary
iter  2 act 7.239e+01 pre 7.937e+01 delta 7.267e+00 f 1.821e+02 |g| 2.427e+02 CG   5
iter  3 act -4.847e+02 pre 5.427e+01 delta 1.356e+00 f 1.097e+02 |g| 1.491e+02 CG  24
cg reaches trust region boundary
iter  3 act 3.406e+01 pre 3.463e+01 delta 5.424e+00 f 1.097e+02 |g| 1.491e+02 CG   7
cg reaches trust region boundary
iter  4 act -7.534e+02 pre 2.447e+01 delta 1.356e+00 f 7.562e+01 |g| 4.025e+01 CG  17
cg reaches trust region boundary
iter  4 act -9.935e+00 pre 1.035e+01 delta 4.513e-01 f 7.562e+01 |g| 4.025e+01 CG   7
cg reaches trust region boundary
iter  4 act 4.453e+00 pre 4.883e+00 delta 1.805e+00 f 7.562e+01 |g| 4.025e+01 CG   5
cg reaches trust region boundary
iter  5 act -2.452e+01 pre 8.040e+00 delta 4.513e-01 f 7.117e+01 |g| 3.584e+01 CG   9
cg reaches trust region boundary
iter  5 act 2.431e+00 pre 3.020e+00 delta 1.805e+00 f 7.117e+01 |g| 3.584e+01 CG   5
cg reaches trust region boundary
iter  6 act -5.926e+01 pre 6.720e+00 delta 4.513e-01 f 6.874e+01 |g| 3.026e+01 CG   8
cg reaches trust region boundary
iter  6 act -3.016e-01 pre 2.442e+00 delta 2.120e-01 f 6.874e+01 |g| 3.026e+01 CG   5
cg reaches trust region boundary
iter  6 act 1.267e+00 pre 1.537e+00 delta 8.479e-01 f 6.874e+01 |g| 3.026e+01 CG   4
cg reaches trust region boundary
iter  7 act -6.390e+00 pre 2.506e+00 delta 2.120e-01 f 6.747e+01 |g| 1.653e+01 CG   6
cg reaches trust region boundary
iter  7 act 5.983e-01 pre 8.945e-01 delta 1.700e-01 f 6.747e+01 |g| 1.653e+01 CG   4
cg reaches trust region boundary
iter  8 act 5.387e-01 pre 7.616e-01 delta 1.345e-01 f 6.687e+01 |g| 1.683e+01 CG   4
cg reaches trust region boundary
iter  9 act 3.846e-01 pre 5.750e-01 delta 1.064e-01 f 6.633e+01 |g| 1.961e+01 CG   4
cg reaches trust region boundary
iter 10 act 3.109e-01 pre 4.043e-01 delta 4.257e-01 f 6.595e+01 |g| 1.240e+01 CG   4
cg reaches trust region boundary
iter 11 act -1.999e+00 pre 1.039e+00 delta 1.064e-01 f 6.564e+01 |g| 9.800e+00 CG   5
cg reaches trust region boundary
iter 11 act 1.941e-01 pre 3.207e-01 delta 8.213e-02 f 6.564e+01 |g| 9.800e+00 CG   4
cg reaches trust region boundary
iter 12 act 2.283e-01 pre 2.847e-01 delta 3.285e-01 f 6.544e+01 |g| 1.047e+01 CG   4
Accuracy = 71.0579% (356/501)

## without cv and s=1
(pytorch) ➜  germeval2018 git:(master) ✗ python 00_base_nbsvm.py --liblinear ./liblinear-2.20 --ftrain germeval2018.training.txt --ftest germeval2018.test.txt --out SCORE.txt --ngram 12
counting (train only)..
vocab len 88386
computing r..
processing files..
..*
optimization finished, #iter = 23
Objective value = -96.029910
nSV = 2874
Accuracy = 72.6547% (364/501)


# SimpleRNN

## with text
train len 3507
val len 500
test len 1002
text vocab size 14604
lemma vocab size 13836
label vocab size 2
EPOCH: 01 - TRN_LOSS: 0.656 - TRN_ACC: 62.98% - VAL_LOSS: 0.720 - VAL_ACC: 54.30%
EPOCH: 02 - TRN_LOSS: 0.647 - TRN_ACC: 64.74% - VAL_LOSS: 0.714 - VAL_ACC: 55.86%
EPOCH: 03 - TRN_LOSS: 0.645 - TRN_ACC: 65.67% - VAL_LOSS: 0.709 - VAL_ACC: 58.20%
EPOCH: 04 - TRN_LOSS: 0.642 - TRN_ACC: 65.82% - VAL_LOSS: 0.711 - VAL_ACC: 57.81%
EPOCH: 05 - TRN_LOSS: 0.642 - TRN_ACC: 65.59% - VAL_LOSS: 0.710 - VAL_ACC: 57.81%
TEST_LOSS: 0.703, TEST_ACC: 56.01%

## with lemma
EPOCH: 01 - TRN_LOSS: 0.650 - TRN_ACC: 64.91% - VAL_LOSS: 0.719 - VAL_ACC: 52.73%
EPOCH: 02 - TRN_LOSS: 0.644 - TRN_ACC: 65.75% - VAL_LOSS: 0.712 - VAL_ACC: 52.54%
EPOCH: 03 - TRN_LOSS: 0.642 - TRN_ACC: 65.63% - VAL_LOSS: 0.708 - VAL_ACC: 52.54%
EPOCH: 04 - TRN_LOSS: 0.641 - TRN_ACC: 65.70% - VAL_LOSS: 0.712 - VAL_ACC: 53.12%
EPOCH: 05 - TRN_LOSS: 0.641 - TRN_ACC: 65.62% - VAL_LOSS: 0.713 - VAL_ACC: 52.93%
TEST_LOSS: 0.700, TEST_ACC: 51.85%
EPOCH: 01 - TRN_LOSS: 0.649 - TRN_ACC: 66.14% - VAL_LOSS: 0.711 - VAL_ACC: 55.47%
EPOCH: 02 - TRN_LOSS: 0.646 - TRN_ACC: 65.62% - VAL_LOSS: 0.706 - VAL_ACC: 56.45%
EPOCH: 03 - TRN_LOSS: 0.643 - TRN_ACC: 66.04% - VAL_LOSS: 0.701 - VAL_ACC: 56.84%
EPOCH: 04 - TRN_LOSS: 0.641 - TRN_ACC: 66.11% - VAL_LOSS: 0.703 - VAL_ACC: 57.42%
EPOCH: 05 - TRN_LOSS: 0.642 - TRN_ACC: 65.91% - VAL_LOSS: 0.702 - VAL_ACC: 57.62%
TEST_LOSS: 0.689, TEST_ACC: 56.11%


## loaded vectors into vocab
EPOCH: 01 - TRN_LOSS: 0.652 - TRN_ACC: 64.23% - VAL_LOSS: 0.702 - VAL_ACC: 53.32%
EPOCH: 02 - TRN_LOSS: 0.646 - TRN_ACC: 65.34% - VAL_LOSS: 0.695 - VAL_ACC: 54.10%
EPOCH: 03 - TRN_LOSS: 0.644 - TRN_ACC: 65.79% - VAL_LOSS: 0.690 - VAL_ACC: 55.47%
EPOCH: 04 - TRN_LOSS: 0.641 - TRN_ACC: 66.25% - VAL_LOSS: 0.691 - VAL_ACC: 56.45%
EPOCH: 05 - TRN_LOSS: 0.641 - TRN_ACC: 65.96% - VAL_LOSS: 0.690 - VAL_ACC: 56.64%
TEST_LOSS: 0.677, TEST_ACC: 58.49%

## used vectors actually in RNN ;-)
EPOCH: 01 - TRN_LOSS: 0.689 - TRN_ACC: 53.89% - VAL_LOSS: 0.649 - VAL_ACC: 62.50%
EPOCH: 02 - TRN_LOSS: 0.671 - TRN_ACC: 65.74% - VAL_LOSS: 0.630 - VAL_ACC: 63.87%
EPOCH: 03 - TRN_LOSS: 0.659 - TRN_ACC: 66.12% - VAL_LOSS: 0.622 - VAL_ACC: 64.84%
EPOCH: 04 - TRN_LOSS: 0.652 - TRN_ACC: 66.39% - VAL_LOSS: 0.619 - VAL_ACC: 64.84%
EPOCH: 05 - TRN_LOSS: 0.647 - TRN_ACC: 66.42% - VAL_LOSS: 0.621 - VAL_ACC: 65.04%
TEST_LOSS: 0.626, TEST_ACC: 65.44%

## more data
train len 4007
val len 1002
test len 3398
text vocab size 15951
lemma vocab size 15100
label vocab size 2
EPOCH: 01 - TRN_LOSS: 0.694 - TRN_ACC: 56.92% - VAL_LOSS: 0.701 - VAL_ACC: 50.36%
EPOCH: 02 - TRN_LOSS: 0.669 - TRN_ACC: 65.71% - VAL_LOSS: 0.661 - VAL_ACC: 63.06%
EPOCH: 03 - TRN_LOSS: 0.657 - TRN_ACC: 66.29% - VAL_LOSS: 0.651 - VAL_ACC: 63.15%
EPOCH: 04 - TRN_LOSS: 0.648 - TRN_ACC: 66.33% - VAL_LOSS: 0.648 - VAL_ACC: 63.15%
EPOCH: 05 - TRN_LOSS: 0.643 - TRN_ACC: 66.35% - VAL_LOSS: 0.647 - VAL_ACC: 63.25%
TEST_LOSS: 0.647, TEST_ACC: 66.35%


## LSTM, bidirectional=True, dropout 0.3
EPOCH: 01 - TRN_LOSS: 0.622 - TRN_ACC: 66.44% - VAL_LOSS: 0.612 - VAL_ACC: 66.59%
EPOCH: 02 - TRN_LOSS: 0.539 - TRN_ACC: 72.20% - VAL_LOSS: 0.539 - VAL_ACC: 71.90%
EPOCH: 03 - TRN_LOSS: 0.449 - TRN_ACC: 78.73% - VAL_LOSS: 0.546 - VAL_ACC: 72.32%
EPOCH: 04 - TRN_LOSS: 0.342 - TRN_ACC: 85.00% - VAL_LOSS: 0.572 - VAL_ACC: 72.52%
EPOCH: 05 - TRN_LOSS: 0.262 - TRN_ACC: 89.30% - VAL_LOSS: 0.620 - VAL_ACC: 71.23%
EPOCH: 06 - TRN_LOSS: 0.196 - TRN_ACC: 92.33% - VAL_LOSS: 0.685 - VAL_ACC: 74.94%
EPOCH: 07 - TRN_LOSS: 0.161 - TRN_ACC: 93.58% - VAL_LOSS: 0.889 - VAL_ACC: 68.17%
EPOCH: 08 - TRN_LOSS: 0.119 - TRN_ACC: 95.36% - VAL_LOSS: 0.745 - VAL_ACC: 75.54%
EPOCH: 09 - TRN_LOSS: 0.104 - TRN_ACC: 95.84% - VAL_LOSS: 1.082 - VAL_ACC: 68.13%
EPOCH: 10 - TRN_LOSS: 0.088 - TRN_ACC: 96.66% - VAL_LOSS: 0.894 - VAL_ACC: 70.04%
TEST_LOSS: 1.039, TEST_ACC: 65.71%


## LSTM, bidirectional=True, dropout 0.4
EPOCH: 00 - TRN_LOSS: 0.625 - TRN_ACC: 66.39% - VAL_LOSS: 0.599 - VAL_ACC: 66.15%
EPOCH: 01 - TRN_LOSS: 0.552 - TRN_ACC: 71.16% - VAL_LOSS: 0.548 - VAL_ACC: 71.51%
EPOCH: 02 - TRN_LOSS: 0.493 - TRN_ACC: 75.96% - VAL_LOSS: 0.530 - VAL_ACC: 71.61%
EPOCH: 03 - TRN_LOSS: 0.395 - TRN_ACC: 82.11% - VAL_LOSS: 0.573 - VAL_ACC: 71.07%
EPOCH: 04 - TRN_LOSS: 0.312 - TRN_ACC: 86.86% - VAL_LOSS: 0.573 - VAL_ACC: 74.11%
EPOCH: 05 - TRN_LOSS: 0.251 - TRN_ACC: 89.47% - VAL_LOSS: 0.651 - VAL_ACC: 72.76%
EPOCH: 06 - TRN_LOSS: 0.213 - TRN_ACC: 91.41% - VAL_LOSS: 0.755 - VAL_ACC: 69.76%
TEST_LOSS: 0.848, TEST_ACC: 65.45%


## LSTM, bidirectional=True, dropout 0.5 (only 0.05 validation)
EPOCH: 00 - TRN_LOSS: 0.621 - TRN_ACC: 66.01% - VAL_LOSS: 0.660 - VAL_ACC: 58.44%
EPOCH: 01 - TRN_LOSS: 0.563 - TRN_ACC: 70.65% - VAL_LOSS: 0.657 - VAL_ACC: 65.55%
EPOCH: 02 - TRN_LOSS: 0.506 - TRN_ACC: 74.66% - VAL_LOSS: 0.605 - VAL_ACC: 67.50%
EPOCH: 03 - TRN_LOSS: 0.436 - TRN_ACC: 79.31% - VAL_LOSS: 0.605 - VAL_ACC: 72.27%
EPOCH: 04 - TRN_LOSS: 0.365 - TRN_ACC: 83.55% - VAL_LOSS: 0.572 - VAL_ACC: 70.86%
EPOCH: 05 - TRN_LOSS: 0.310 - TRN_ACC: 86.70% - VAL_LOSS: 0.656 - VAL_ACC: 69.14%
EPOCH: 06 - TRN_LOSS: 0.267 - TRN_ACC: 88.32% - VAL_LOSS: 0.649 - VAL_ACC: 73.67%
TEST_LOSS: 0.701, TEST_ACC: 70.50%




EPOCH: 00 - TRN_LOSS: 0.628 - TRN_ACC: 66.17% - VAL_LOSS: 0.637 - VAL_ACC: 1887.50% - VAL_REC: 0.4405 - VAL_PRE: 0.4157 - VAL_F1: 0.4277
EPOCH: 01 - TRN_LOSS: 0.579 - TRN_ACC: 68.96% - VAL_LOSS: 0.617 - VAL_ACC: 2087.50% - VAL_REC: 0.4405 - VAL_PRE: 0.5068 - VAL_F1: 0.4713
EPOCH: 02 - TRN_LOSS: 0.539 - TRN_ACC: 72.84% - VAL_LOSS: 0.602 - VAL_ACC: 2187.50% - VAL_REC: 0.5119 - VAL_PRE: 0.5584 - VAL_F1: 0.5342
EPOCH: 03 - TRN_LOSS: 0.512 - TRN_ACC: 74.43% - VAL_LOSS: 0.590 - VAL_ACC: 2225.00% - VAL_REC: 0.7619 - VAL_PRE: 0.5517 - VAL_F1: 0.6400
EPOCH: 04 - TRN_LOSS: 0.450 - TRN_ACC: 78.86% - VAL_LOSS: 0.577 - VAL_ACC: 2225.00% - VAL_REC: 0.3333 - VAL_PRE: 0.6364 - VAL_F1: 0.4375
EPOCH: 05 - TRN_LOSS: 0.383 - TRN_ACC: 82.92% - VAL_LOSS: 0.598 - VAL_ACC: 2262.50% - VAL_REC: 0.6667 - VAL_PRE: 0.5773 - VAL_F1: 0.6188
EPOCH: 06 - TRN_LOSS: 0.335 - TRN_ACC: 85.06% - VAL_LOSS: 0.610 - VAL_ACC: 2275.00% - VAL_REC: 0.6667 - VAL_PRE: 0.5833 - VAL_F1: 0.6222
TEST_LOSS: 0.674, TEST_ACC: 2117.76% - TEST_REC: 0.5739 - TEST_PRE: 0.5069 - TEST_F1: 0.5383


EPOCH: 00
TRN_LOS: 0.628 - TRN_ACC: 66.17% - VAL_LOS: 0.637 - VAL_ACC: 60.40% - REC: 0.4405 - PRE: 0.4157 - F1: 0.4277
EPOCH: 01
TRN_LOS: 0.579 - TRN_ACC: 68.96% - VAL_LOS: 0.617 - VAL_ACC: 66.80% - REC: 0.4405 - PRE: 0.5068 - F1: 0.4713
EPOCH: 02
TRN_LOS: 0.539 - TRN_ACC: 72.84% - VAL_LOS: 0.602 - VAL_ACC: 70.00% - REC: 0.5119 - PRE: 0.5584 - F1: 0.5342
EPOCH: 03
TRN_LOS: 0.512 - TRN_ACC: 74.43% - VAL_LOS: 0.590 - VAL_ACC: 71.20% - REC: 0.7619 - PRE: 0.5517 - F1: 0.6400
EPOCH: 04
TRN_LOS: 0.450 - TRN_ACC: 78.86% - VAL_LOS: 0.577 - VAL_ACC: 71.20% - REC: 0.3333 - PRE: 0.6364 - F1: 0.4375
EPOCH: 05
TRN_LOS: 0.383 - TRN_ACC: 82.92% - VAL_LOS: 0.598 - VAL_ACC: 72.40% - REC: 0.6667 - PRE: 0.5773 - F1: 0.6188
EPOCH: 06
TRN_LOS: 0.335 - TRN_ACC: 85.06% - VAL_LOS: 0.610 - VAL_ACC: 72.80% - REC: 0.6667 - PRE: 0.5833 - F1: 0.6222
TEST_LOS: 0.674, TEST_ACC: 66.69% - TEST_REC: 0.5739 - TEST_PRE: 0.5069 - TEST_F1: 0.5383


## LSTM, bidirectional=True, dropout 0.6 (only 0.05 validation)
EPOCH: 00
TRN_LOS: 0.636 - TRN_ACC: 66.19% - VAL_LOS: 0.612 - VAL_ACC: 67.20% - REC: 0.0238 - PRE: 1.0000 - F1: 0.0465
EPOCH: 01
TRN_LOS: 0.599 - TRN_ACC: 67.61% - VAL_LOS: 0.639 - VAL_ACC: 62.00% - REC: 0.7024 - PRE: 0.4574 - F1: 0.5540
EPOCH: 02
TRN_LOS: 0.567 - TRN_ACC: 69.92% - VAL_LOS: 0.609 - VAL_ACC: 65.20% - REC: 0.7262 - PRE: 0.4880 - F1: 0.5837
EPOCH: 03
TRN_LOS: 0.537 - TRN_ACC: 72.11% - VAL_LOS: 0.667 - VAL_ACC: 58.40% - REC: 0.7976 - PRE: 0.4351 - F1: 0.5630
EPOCH: 04
TRN_LOS: 0.499 - TRN_ACC: 75.47% - VAL_LOS: 0.563 - VAL_ACC: 71.20% - REC: 0.5833 - PRE: 0.5698 - F1: 0.5765
EPOCH: 05
TRN_LOS: 0.452 - TRN_ACC: 78.87% - VAL_LOS: 0.614 - VAL_ACC: 71.60% - REC: 0.6905 - PRE: 0.5631 - F1: 0.6203
EPOCH: 06
TRN_LOS: 0.401 - TRN_ACC: 81.68% - VAL_LOS: 0.768 - VAL_ACC: 62.80% - REC: 0.8690 - PRE: 0.4710 - F1: 0.6109
TEST_LOS: 0.821, TEST_ACC: 54.18% - TEST_REC: 0.8183 - TEST_PRE: 0.4111 - TEST_F1: 0.5473


## LSTM, bidirectional=True, dropout 0.6 (only 0.05 validation) HID_SIZE=80, NUM_RNN=2
EPOCH: 00
TRN_LOS: 0.643 - TRN_ACC: 66.38% - VAL_LOS: 0.632 - VAL_ACC: 66.40% - REC: 0.0000 - PRE: 0.0000 - F1: 0.0000
EPOCH: 01
TRN_LOS: 0.623 - TRN_ACC: 66.00% - VAL_LOS: 0.663 - VAL_ACC: 60.80% - REC: 0.2024 - PRE: 0.3542 - F1: 0.2576
EPOCH: 02
TRN_LOS: 0.591 - TRN_ACC: 68.78% - VAL_LOS: 0.620 - VAL_ACC: 65.60% - REC: 0.5357 - PRE: 0.4891 - F1: 0.5114
EPOCH: 03
TRN_LOS: 0.581 - TRN_ACC: 69.01% - VAL_LOS: 0.621 - VAL_ACC: 65.20% - REC: 0.6905 - PRE: 0.4874 - F1: 0.5714
EPOCH: 04
TRN_LOS: 0.556 - TRN_ACC: 70.84% - VAL_LOS: 0.589 - VAL_ACC: 67.20% - REC: 0.4048 - PRE: 0.5152 - F1: 0.4533
EPOCH: 05
TRN_LOS: 0.534 - TRN_ACC: 72.64% - VAL_LOS: 0.600 - VAL_ACC: 65.20% - REC: 0.7143 - PRE: 0.4878 - F1: 0.5797
EPOCH: 06
TRN_LOS: 0.507 - TRN_ACC: 75.00% - VAL_LOS: 0.651 - VAL_ACC: 68.80% - REC: 0.7619 - PRE: 0.5246 - F1: 0.6214
TEST_LOS: 0.635, TEST_ACC: 66.54% - TEST_REC: 0.6443 - TEST_PRE: 0.5044 - TEST_F1: 0.5659

## EMB_SIZE=100 HID_SIZE=70 DROPOUT=0.7
EPOCH: 00
TRN_LOS: 0.648 - TRN_ACC: 65.99% - VAL_LOS: 0.643 - VAL_ACC: 66.40% - REC: 0.0000 - PRE: 0.0000 - F1: 0.0000
EPOCH: 01
TRN_LOS: 0.635 - TRN_ACC: 66.26% - VAL_LOS: 0.591 - VAL_ACC: 66.80% - REC: 0.0357 - PRE: 0.6000 - F1: 0.0674
EPOCH: 02
TRN_LOS: 0.618 - TRN_ACC: 66.11% - VAL_LOS: 0.565 - VAL_ACC: 70.40% - REC: 0.3333 - PRE: 0.6087 - F1: 0.4308
EPOCH: 03
TRN_LOS: 0.598 - TRN_ACC: 68.57% - VAL_LOS: 0.563 - VAL_ACC: 69.20% - REC: 0.6429 - PRE: 0.5347 - F1: 0.5838
EPOCH: 04
TRN_LOS: 0.583 - TRN_ACC: 69.27% - VAL_LOS: 0.530 - VAL_ACC: 72.00% - REC: 0.4643 - PRE: 0.6094 - F1: 0.5270
EPOCH: 05
TRN_LOS: 0.561 - TRN_ACC: 71.07% - VAL_LOS: 0.580 - VAL_ACC: 67.60% - REC: 0.8571 - PRE: 0.5106 - F1: 0.6400
EPOCH: 06
TRN_LOS: 0.535 - TRN_ACC: 72.74% - VAL_LOS: 0.539 - VAL_ACC: 72.80% - REC: 0.8452 - PRE: 0.5635 - F1: 0.6762
TEST_LOS: 0.596, TEST_ACC: 68.25% - TEST_REC: 0.5522 - TEST_PRE: 0.5296 - TEST_F1: 0.5407

## EMB_SIZE=100 HID_SIZE=70 DROPOUT=0.7 FULL-DATA-SET
train len 5885
val len 1682
test len 840
EPOCH: 00
TRN_LOS: 0.645 - TRN_ACC: 65.96% - VAL_LOS: 0.722 - VAL_ACC: 66.23% - REC: 0.0000 - PRE: 0.0000 - F1: 0.0000
EPOCH: 01
TRN_LOS: 0.636 - TRN_ACC: 66.29% - VAL_LOS: 0.606 - VAL_ACC: 66.29% - REC: 0.0018 - PRE: 1.0000 - F1: 0.0035
EPOCH: 02
TRN_LOS: 0.607 - TRN_ACC: 67.15% - VAL_LOS: 0.590 - VAL_ACC: 66.88% - REC: 0.0475 - PRE: 0.6279 - F1: 0.0884
EPOCH: 03
TRN_LOS: 0.589 - TRN_ACC: 68.92% - VAL_LOS: 0.602 - VAL_ACC: 67.42% - REC: 0.2482 - PRE: 0.5382 - F1: 0.3398
EPOCH: 04
TRN_LOS: 0.571 - TRN_ACC: 70.57% - VAL_LOS: 0.563 - VAL_ACC: 69.14% - REC: 0.5158 - PRE: 0.5456 - F1: 0.5303
EPOCH: 05
TRN_LOS: 0.548 - TRN_ACC: 71.88% - VAL_LOS: 0.576 - VAL_ACC: 67.42% - REC: 0.7148 - PRE: 0.5126 - F1: 0.5971
EPOCH: 06
TRN_LOS: 0.509 - TRN_ACC: 74.43% - VAL_LOS: 0.562 - VAL_ACC: 71.46% - REC: 0.6673 - PRE: 0.5657 - F1: 0.6123
TEST_LOS: 0.528, TEST_ACC: 75.71% - TEST_REC: 0.7067 - TEST_PRE: 0.6231 - TEST_F1: 0.6623

## EMB_SIZE=100 HID_SIZE=70 DROPOUT=0.7 FULL-DATA-SET (0.1 val + test)
train len 6725
val len 841
test len 841
text vocab size 22115
lemma vocab size 20770
EPOCH: 00
TRN_LOS: 0.642 - TRN_ACC: 66.18% - VAL_LOS: 0.635 - VAL_ACC: 66.23% - REC: 0.0000 - PRE: 0.0000 - F1: 0.0000
EPOCH: 01
TRN_LOS: 0.627 - TRN_ACC: 66.29% - VAL_LOS: 0.576 - VAL_ACC: 67.42% - REC: 0.0669 - PRE: 0.6786 - F1: 0.1218
EPOCH: 02
TRN_LOS: 0.599 - TRN_ACC: 67.12% - VAL_LOS: 0.543 - VAL_ACC: 71.58% - REC: 0.3944 - PRE: 0.6257 - F1: 0.4838
EPOCH: 03
TRN_LOS: 0.568 - TRN_ACC: 70.76% - VAL_LOS: 0.600 - VAL_ACC: 64.21% - REC: 0.8944 - PRE: 0.4838 - F1: 0.6279
EPOCH: 04
TRN_LOS: 0.554 - TRN_ACC: 70.92% - VAL_LOS: 0.532 - VAL_ACC: 74.55% - REC: 0.6338 - PRE: 0.6207 - F1: 0.6272
EPOCH: 05
TRN_LOS: 0.524 - TRN_ACC: 74.71% - VAL_LOS: 0.529 - VAL_ACC: 72.53% - REC: 0.8063 - PRE: 0.5654 - F1: 0.6647
EPOCH: 06
TRN_LOS: 0.487 - TRN_ACC: 76.21% - VAL_LOS: 0.534 - VAL_ACC: 72.41% - REC: 0.7430 - PRE: 0.5703 - F1: 0.6453
TEST_LOS: 0.546, TEST_ACC: 70.51% - TEST_REC: 0.7465 - TEST_PRE: 0.5464 - TEST_F1: 0.6310

## 