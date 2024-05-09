# Machine Learning: KNN in Rust

My series of Machine Learning experiences in Rust; and first in my list is the hello-world of ML _classifications_ - [K-Nearest Neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithm

Truth be told, main inspirations is because of this gentleman's video: [clumsy computer](https://www.youtube.com/watch?v=vzabeKdW9tE) which is quite entertaining to watch, mainly because we've all done the tensorflow tutorial of [Basic image classification](https://www.tensorflow.org/tutorials/keras/classification) and were left wondering what it was all about... Another side note is that the gentleman's video is about using [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) and calling KNN algorithm to predict its classifications by taking the top K predictions and counting the most encountered.

## Some differences

One thing that inspired me into trying this out was (as mentioned above of the video that inspired me) KNN is slow, and what I wanted to experiement was on how I can speed up based on parallelizing the comparison against the trained images. MNIST database consits of 60K images, so with 4 threads, I can break that down to 15K per thread, and so on.

Initial thinking, because image comparison (at least, my initial intents) will be based on flattening the 2D image into 1D, and comparing byte-by-byte (`uint8`), if the image is 28x28 pixels (flatten to 784 pixels), that's 784 comparions per trained image. I've watched the video in which the coder prints out the index of the trained image it was comparing against, and the index he's print out would show in rate of about 1 per second... The coder also converted a single byte (`uint8`) into a 32-bit (`uint32`), wasting 24-bits per read.

Note: In C++, that's `std::uint8_t` and `std::uint32_t`; whether in C++ or Rust, I will stay away from using the term "word" and "long", though no matter which language, "byte" is 8-bits, due to ambiguities in languages and native settings, such as `size_t` and `int` can be 32 or 64 bits, or to express 64-bits as "`long long`", `char` can be 8, 16, or 32-bits compared to `wchar` being platform specfic to Windows for 16-bit, etc, is just TOO AMBIGUOUS! 30+ years ago, we had to also consider using C libraries like `htonl()` to deal with byte-sex... Sorry for the rant, but this is one of the things about Rust, the native types are explicit in that way and static-analyzer will even nag (and hate you and haunt your underwear drawers until you fix it or dynamically cast it) if you try to assign `uint32` into an `int32` variable (and vice-versa)!

With that said, when doing native comparison, the byte-sex (big-end-in vs little-end-in) does not matter, hence I should be able to just read 4 bytes (like a block, similar thinking like mpeg?) directly into the register, and when doing comparison, just do comparison 7 times of 4-bytes (`uint32`)

```rust
  // obviously, we should be packing a 28 bytes x 28 bytes using iteration (for row=0..28{forcol=0..28}) but this is just pseudo-code...
  // row 0, columns 0..27
  auto dst_block_1_1 = dst_image[(0 * 7) + ((0 * 7) + 0)] || dst_image[(0*7) + ((0*7)+1)] << 1 || dst_image[2] << 2 || dst_image[3] << 3
  ...
  auto dst_block_1_7 = dst_image[(0 * 7) + ((4 * 7) + 0)] || dst_image[(0*7) + ((4*7)+1)] << 1 || dst_image[26] << 2 || dst_image[27] << 3
  ...
  // row 7, columns 0..27
  auto dst_block_7_7 = dst_image[(7 * 7) + ((4 * 7) + 0)] || dst_image[(7*7) + ((4*7)+1)] << 1 || dst_image[(7*7)+26] << 2 || dst_image[(7*7)+27] << 3
for (image_index, image) in images.enumerate() {
  // similar to above, should iterate rowxcol but for demonstration,
  auto block_1_1 = image[image_index + 0] || image[image_index + 1] << 1 || image[image_index + 2] << 2 || image[image_index + 3] << 3
  ...
  auto block_1_7 = image[image_index + 24] || image[image_index + 25] << 1 || image[image_index + 26] << 2 || image[image_index + 27] << 3
  ...

  // from experiences, it's better to do XOR(^) instead of AND(&) for image comparison
  auto result_1_1 = dst_block_1_1 ^ block_1_1
  ...
  auto resutl_1_7 = dst_block_1_7 ^ block_1_7;
  ...
}
```

Of course, comparing byte-by-byte versus block-by-block probably differs in some cases - i.e. comparing `0xFF00FF00` XOR `0x00FF00FF` (XOR:`0xFFffFFff`, AND:`0x00000000`) is basically, if I had shifted 1 pixel to the left, it may have matched but because I was greedy to compare by blocks I have completely caused false-negatives. Side note mentioned below, is that this is assuming I'm doing text-OCR (numbers) instead of clothes database, in which comparing `0xEE00EE00` XOR `0xFF00FF00` (XOR:`0x11001100`, AND: `0xEE00EE00`).

Note also that if we're going that route of packing 4 pixels into a 32-bit register, why not pack each pixels into a bit, so each row can fit in 28bits, resulting in 7 32-bits `uint32`'s? That solution seems ideal if the pixels were either lit (enabled) or unlit (disabled), and one can probably test for greyscales of values bigger than `0x7F` to be enabled. This (probably) will work well for text-OCR in which texts are on a white background (note that MNIST database for number datasets uses white background with black numbers, so that's reversed from what I'm saying), but it'll probably (definitely) fail on the clothes database making it super difficult to differentiate the sillouette of t-shirt and coat.

All in all, I'd like to try out few matrix, but first with following assumptions:

- training data are split into N threads (i.e. N=4, at 60K, each thread only needs to iterate-compare with 15K)
- k is constant, and for each test/trials, it will use same same constant throught

Here are the matrix:

- into_bytes: read data as byte[] (single dimensional) array, and straight-forward comparision
- into_bytes: read data as byte[] (single dimensional) array, compare in parallel
- into_32bits: reads 4-bytes at a time and store it as `uint32` arrays, and again straight forward comparision by 4 bytes

## Technology

If you follow the breadcrumb chain from:

1. [Tensorflow classification tutorial](https://www.tensorflow.org/tutorials/keras/classification)
2. click on [Fashion MINIST](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb#scrollTo=dzLKpmZICaWN) in the dataset section
3. which takes you to [fashion-mnist on github](https://github.com/zalandoresearch/fashion-mnist)
4. in which if you scroll down to the bottom of the language section for Rust, it takes you to [vsion-rs](https://github.com/AtheMathmo/vision-rs/tree/master)
5. in which takes you to the [vision crate](https://crates.io/crates/vision)

Long story short, I'm using "vision" crate because I want to start with data as-is (i.e. similar to using `include_bytes!` macro); From there, I'll init/setup transposing from `vec<vec<u8>>` into() `vec<uint32>` and `vec<u8>`.

NOTE that dataset from lecan.com no longer seems to be available at the time of this writing, and annoyingly, the [zalandorresearch](https://github.com/zalandoresearch/fashion-mnist) downloads are in EU and even though the data are small, I keep timing out. I tricked keras (on tensorflow tutorials for image classifications) to load the dataset which will log the URL of where the datasets were download from, in which I copied and did `wget` for each (4) .gz files and copied it to the "data_home()" paths, in which the library is clever enough to log as "Already downloaded" and go on its way... Note that unit-test for the crate uses "MNIST" as the dir-home, probably good idea to just hand-download to MNIST dir :shrug:

For distance algorithms, for now, I'm only implementing Euclidean, but if time is avail, I'll also integrate Hamming.

Here are the current `Config` setup for args:

```rust
// --data_home - path to the MNIST data
// --verbose - print debug info
// --k - number of neighbors to consider
// --algorithm - distance algorithm to use (euclidean or hamming)
// --byte_size - number of bytes to use for each pixel (1, 4, or 8)
// Default: --data_home=data/MNIST --verbose --k=3 --algorithm=euclidean --byte_size=1
```

```bash
Creating data directory: data/MNIST
Already downloaded
Extracting data
MNIST Loaded!
train_imgs len=60000
test_imgs len=10000
train_labels len=60000
test_labels len=10000
k=3
algorithm=Euclidean
byte_size=One
accuracy=0.8541
elapsed time=2366.8142956s  <-- that's about 40 minutes!
```

Note that above sample output is using single thread to compare each test image 60,000 times (in release build)!

## Postmortem and Caveats

- I want to get started on this today so I'll come back to this section later...
