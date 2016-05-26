# examples/convolutional_vae.py
# TODO re-do numbers here

# Run with 1 update per epoch:
# python convolutional_vae.py --n_iter_per_epoch=1
# Check that objective values are as follows:
-0.831284
-0.834811
-0.783949
-0.689275
-0.653788
-0.656840
-0.650696
-0.622813
-0.619293
-0.608524

# Run with 1000 updates per epoch:
# python convolutional_vae.py --n_iter_per_epoch=1000
# Check that objective values are as follows:
-0.219641
-0.143808

# Check that generated images after 2 epochs look like digits.
