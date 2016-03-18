# examples/convolutional_vae.py

# Run with 1 update per epoch:
# python convolutional_vae.py --n_iter_per_epoch=1
# Check that objective values are as follows:
0.827012
0.886294
0.792136
0.668022
0.660516
0.668974
0.648113
0.615716
0.616493
0.612958

# Run with 1000 updates per epoch:
# python convolutional_vae.py --n_iter_per_epoch=1000
# Check that objective values are as follows:
0.219992
0.144405
0.137927
0.134993

# Check that generated images after 2 epochs look like digits.
