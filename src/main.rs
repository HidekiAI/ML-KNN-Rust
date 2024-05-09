extern crate vision;

use vision::mnist::MNISTBuilder;

const THREAD_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
enum DistanceAlgorithm {
    Euclidean,
    Hamming,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum ByteSize {
    One,
    Four,
    Eight,
}

#[derive(Clone, Debug)]
struct Config {
    data_home: String,
    verbose: bool,
    k: usize,
    algorithm: DistanceAlgorithm,
    byte_size: ByteSize,
}
impl Default for Config {
    fn default() -> Config {
        Config {
            data_home: "data/MNIST".to_string(),
            verbose: false,
            k: 3,
            algorithm: DistanceAlgorithm::Euclidean,
            byte_size: ByteSize::One,
        }
    }
}

fn get_distance_euclidean(a: &[u8], b: &[u8]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] as f64 - b[i] as f64).powi(2);
    }
    sum.sqrt()
}

fn get_distance_hamming(a: &[u8], b: &[u8]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] ^ b[i]).count_ones() as f64;
    }
    sum
}

fn get_k_nearest_neighbors(
    train_imgs: &Vec<Vec<u8>>,
    test_img: &Vec<u8>,
    k: &usize,
    alg: &DistanceAlgorithm,
) -> Vec<usize> {
    let mut distances = Vec::new();

    //for (i, train_img) in train_imgs.iter().enumerate() {
    //    let distance = get_distance_euclidean(train_img, test_img);
    //    distances.push((i, distance));
    //}
    //distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    //distances.iter().take(*k).map(|x| x.0).collect()

    // If there are 60,000 trained images, splitting it up into 4 threads, there will be 15,000 images per thread
    let chunk_size = train_imgs.len() / THREAD_COUNT;
    let mut threads = Vec::new();

    // pre-slice the data into chunks outside of the thread-loop so that each thread
    // does not need to clone (deep-copy) the original trained data since it can get expensive
    // memory-wise as well as performance-wise
    let mut trained_img_slices = Vec::new();
    for thread_index in 0..THREAD_COUNT {
        let start = thread_index * chunk_size;
        let end = if thread_index == THREAD_COUNT - 1 {
            train_imgs.len()
        } else {
            (thread_index + 1) * chunk_size
        };
        // slice (meta data) of the read-only trained data
        trained_img_slices.push(train_imgs[start..end].to_vec());
    }

    // NOTE: Because labels are basically index
    for thread_index in 0..THREAD_COUNT {
        // snapshot/clone the (read-only) data to be used in each threads
        let trained_imgs_slice = trained_img_slices[thread_index].clone(); // TODO: Explain WHY I had to clone this to silence the spawn(move||)...
        let test_img_cloned = test_img.clone();
        let distances_cloned = distances.clone();
        let alg_cloned = alg.clone();
        // Because labels are basically index, we need to adjust the label index
        let trained_img_slice_start_index = thread_index * chunk_size;

        // fork (using std::thread instead of tokio since we are not doing async I/O here)
        threads.push(std::thread::spawn(move || {
            let mut distances = distances_cloned;
            for (trained_image_index, train_img) in trained_imgs_slice.iter().enumerate() {
                let distance = match alg_cloned {
                    DistanceAlgorithm::Euclidean => {
                        get_distance_euclidean(train_img, &test_img_cloned)
                    }
                    DistanceAlgorithm::Hamming => get_distance_hamming(train_img, &test_img_cloned),
                };
                distances.push((trained_image_index + trained_img_slice_start_index, distance));
            }
            distances
        }));
    }

    // wait for all to forks finish/rejoin
    for thread in threads {
        let result = thread.join().unwrap();
        distances.extend(result);
    }
    // gather/collect results from each thread
    // distances is tuple of:
    // tup.0 - index of the trained image
    // tup.1 - distance between the trained image and the test image
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(*k).map(|x| x.0).collect()
}

fn get_majority_label(labels: &Vec<u8>, neighbors: &Vec<usize>) -> u8 {
    let mut counts = [0; 10];
    for &neighbor in neighbors {
        counts[labels[neighbor] as usize] += 1;
    }
    let mut max_count = 0;
    let mut max_label = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            max_label = i;
        }
    }
    max_label as u8
}

fn get_accuracy(
    train_imgs: &Vec<Vec<u8>>,
    train_labels: &Vec<u8>,
    test_imgs: &Vec<Vec<u8>>,
    test_labels: &Vec<u8>,
    config: &Config,
) -> f64 {
    let k = &config.k;
    let mut correct = 0;
    for (i, test_img) in test_imgs.iter().enumerate() {
        let neighbors = get_k_nearest_neighbors(train_imgs, test_img, k, &config.algorithm);
        let label = get_majority_label(train_labels, &neighbors);
        if label == test_labels[i] {
            correct += 1;
        }
    }
    correct as f64 / test_imgs.len() as f64
}

// --data_home - path to the MNIST data
// --verbose - print debug info
// --k - number of neighbors to consider
// --algorithm - distance algorithm to use (euclidean or hamming)
// --byte_size - number of bytes to use for each pixel (1, 4, or 8)
// Default: --data_home=data/MNIST --verbose --k=3 --algorithm=euclidean --byte_size=1
fn get_config(args: &Vec<String>) -> Config {
    let mut data_home = "data/MNIST".to_string();
    let mut verbose = false;
    let mut k = 3;
    let mut algorithm = DistanceAlgorithm::Euclidean;
    let mut byte_size = ByteSize::One;
    for arg in args {
        if arg.starts_with("--data_home=") {
            data_home = arg.split("=").collect::<Vec<&str>>()[1].to_string();
        } else if arg == "--verbose" {
            verbose = true;
        } else if arg.starts_with("--k=") {
            k = arg.split("=").collect::<Vec<&str>>()[1].parse().unwrap();
        } else if arg.starts_with("--algorithm=") {
            let algorithm_str = arg.split("=").collect::<Vec<&str>>()[1];
            algorithm = match algorithm_str {
                "euclidean" => DistanceAlgorithm::Euclidean,
                "hamming" => DistanceAlgorithm::Hamming,
                _ => panic!("Invalid algorithm"),
            };
        } else if arg.starts_with("--byte_size=") {
            let byte_size_str = arg.split("=").collect::<Vec<&str>>()[1];
            byte_size = match byte_size_str {
                "1" => ByteSize::One,
                "4" => ByteSize::Four,
                "8" => ByteSize::Eight,
                _ => panic!("Invalid byte_size"),
            };
        }
    }

    Config {
        data_home: data_home,
        verbose: verbose,
        k: k,
        algorithm: algorithm,
        byte_size: byte_size,
    }
}

fn main() {
    let start_time = std::time::Instant::now();
    let config = get_config(&std::env::args().collect::<Vec<String>>());

    let builder = MNISTBuilder::new();
    let mnist = builder
        .data_home(config.data_home.clone())
        .verbose()
        .get_data()
        .unwrap();
    println!("train_imgs len={}", mnist.train_imgs.len());
    println!("test_imgs len={}", mnist.test_imgs.len());
    println!("train_labels len={}", mnist.train_labels.len());
    println!("test_labels len={}", mnist.test_labels.len());
    println!("k={}", config.k);
    println!("algorithm={:?}", config.algorithm);
    println!("byte_size={:?}", config.byte_size);

    let accuracy = get_accuracy(
        &mnist.train_imgs,
        &mnist.train_labels,
        &mnist.test_imgs,
        &mnist.test_labels,
        &config,
    );
    println!("accuracy={}", accuracy);

    println!("elapsed time={:?}", start_time.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_dir_all;

    #[test]
    #[ignore]
    fn test_builder() {
        let builder = MNISTBuilder::new();
        let mnist = builder.data_home("data/MNIST").get_data().unwrap();
        assert_eq!(mnist.train_imgs.len(), 60000);
        remove_dir_all("data/MNIST").unwrap();
    }

    #[test]
    #[ignore]
    fn test_get_distance() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(get_distance_euclidean(&a, &b), 5.196152422706632);
    }

    #[test]
    #[ignore]
    fn test_get_k_nearest_neighbors() {
        let train_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let test_img = vec![1, 2, 3];
        assert_eq!(
            get_k_nearest_neighbors(&train_imgs, &test_img, &2, &DistanceAlgorithm::Euclidean),
            vec![0, 1]
        );
    }

    #[test]
    #[ignore]
    fn test_get_majority_label() {
        let labels = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let neighbors = vec![0, 1, 2];
        assert_eq!(get_majority_label(&labels, &neighbors), 1);
    }

    #[test]
    #[ignore]
    fn test_get_accuracy() {
        let train_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let train_labels = vec![1, 2, 3];
        let test_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let test_labels = vec![1, 2, 3];
        assert_eq!(
            get_accuracy(
                &train_imgs,
                &train_labels,
                &test_imgs,
                &test_labels,
                &Config::default()
            ),
            1.0
        );
    }
}
