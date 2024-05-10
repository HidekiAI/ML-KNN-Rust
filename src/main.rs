extern crate vision;

use std::{cmp::min, collections::HashMap};

use vision::mnist::MNISTBuilder;

// to avoid 50/50 ties, try to keep the chunk partitions in odd numbers
const MAX_PROCESSES: usize = 5;
const THREAD_COUNT: usize = 3;
const MAX_CLASSIFICATION: usize = 10; // there are 10 classes of fashion items...
const CLASSES: [&str; MAX_CLASSIFICATION] = [
    // this is only useful for unit-test debugging
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

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
    verbose: bool, // good for unit-test
    k: usize,
    algorithm: DistanceAlgorithm,
    byte_size: ByteSize,
    max_process_threads: usize,
    max_nearest_evaluation_threads: usize,
}
impl Default for Config {
    fn default() -> Config {
        let verbose = if cfg!(debug_assertions) { true } else { false };
        Config {
            data_home: "data/MNIST".to_string(),
            verbose: verbose,
            k: 3,
            algorithm: DistanceAlgorithm::Euclidean,
            byte_size: ByteSize::One,
            max_process_threads: MAX_PROCESSES,
            max_nearest_evaluation_threads: THREAD_COUNT,
        }
    }
}

type TPixels8bpp = Vec<u8>; // bitmap, 8-bits per pixel
type TClassification = usize; // it's actually the INDEX to classification table
type TImageID = usize;

// Zip the train_imgs and train_labels into a single struct so that
// concerns about index of the train_imgs and train_labels are not
// needed to be worried about
#[derive(Clone, Debug, PartialEq)]
struct MyImage {
    id: TImageID, // because we split the trained images into blocks/chunks, it's hard to figure out the index of the trained image, so we use this ID to indicate the actual index
    label: TClassification, // it's actually the INDEX to classification table
    pixels: TPixels8bpp, // bitmap, 8-bits per pixel
}
impl MyImage {
    // NOTE: MNIST data are all u8 byte-oriented
    fn zip(labels: Vec<u8>, images: Vec<Vec<u8>>) -> Vec<MyImage> {
        // make sure they align on sizes, unlike other "zip" methods where if rhs is smaller, it'll truncate
        if labels.len() != images.len() {
            panic!(
                "ERROR: labels (len = {}) and images (len = {}) must have the same length",
                labels.len(),
                images.len()
            );
        }
        let mut ret = Vec::new();
        for (index, (label, pixels)) in labels
            .into_iter()
            .map(|l| l as TClassification)
            .zip(images.into_iter())
            .enumerate()
        {
            ret.push(MyImage {
                id: index as TImageID, // when grouped as a whole, index is useless, but for trained images broken down to chunks to process in parallel, it becomes useful
                label,
                pixels,
            });
        }
        ret
    }
}

// https://en.wikipedia.org/wiki/Euclidean_distance
// d(a,b) = sqrt((a-b)^2)
fn get_distance_euclidean(a: &MyImage, b: &MyImage) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.pixels.len() {
        sum += (a.pixels[i] as f64 - b.pixels[i] as f64).powi(2);
    }
    sum.sqrt()
}

// https://en.wikipedia.org/wiki/Hamming_distance
// number of positions at which the corresponding symbols differs
fn get_distance_hamming(a: &MyImage, b: &MyImage) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.pixels.len() {
        sum += (a.pixels[i] ^ b.pixels[i]).count_ones() as f64;
    }
    sum
}

fn get_k_nearest_neighbors(
    trained_images: &Vec<MyImage>,
    test_img: &MyImage,
    config: &Config,
) -> Vec<TImageID> {
    if config.verbose {
        //println!(
        //    "\tget_k_nearest_neighbors(): ThreadID({:?}) - Trained image count: {}",
        //    std::thread::current().id(),
        //    trained_images.len()
        //);
    }
    if trained_images.len() < 1 {
        panic!(
            "ERROR: Unnecessary call to request KNN when there are NO ({}) trained images to evaluate against.",
            trained_images.len()
        );
    }

    // If there are 60,000 trained images, splitting it up into 4 threads, there will be 15,000 images per thread
    let min_thread_count = min(config.max_nearest_evaluation_threads, trained_images.len());
    let chunk_size = trained_images.len() / min_thread_count;
    let mut threads: Vec<std::thread::JoinHandle<Vec<(TImageID, f64)>>> = Vec::new();

    // pre-slice the data into chunks outside of the thread-loop so that each thread
    // does not need to clone (deep-copy) the original trained data since it can get expensive
    // memory-wise as well as performance-wise
    let mut trained_image_slices_sets = Vec::new();
    // problem with using iteration between 0..THREAD_COUNT is that if trained_images.len() < THREAD_COUNT, some of the blocks will be empty!
    for thread_index in 0..min_thread_count {
        let start = thread_index * chunk_size;
        let end = if thread_index == min_thread_count - 1 {
            // NOTE: slicing will allow end+1 even if index does not exist!
            trained_images.len()
        } else {
            min((thread_index + 1) * chunk_size, trained_images.len())
        };

        // slice (meta data) of the read-only trained data
        if start < trained_images.len() && end > start {
            trained_image_slices_sets.push(trained_images[start..end].to_vec());
        }
    }
    if config.verbose {
        //println!("\tThere are sub-count of {} trained images to lookup, breaking it up into {} (INNER) threads (slice blocks of {}), each thread will have {} trained  images",
        //    trained_images.len(), min_thread_count, trained_image_slices_sets.len(), chunk_size );
    }
    if trained_image_slices_sets.len() == 0 {
        panic!("ERROR: There are no trained images");
    }

    // NOTE: Because labels are basically index
    for (slice_index, trained_image_slices) in trained_image_slices_sets.into_iter().enumerate() {
        // snapshot/clone the (read-only) data to be used in each threads
        let test_img_cloned = test_img.clone();
        let mut distances_per_thread = Vec::new();
        let alg_cloned = config.algorithm.clone();
        // Because labels are basically index, we need to adjust the label index
        let _trained_img_slice_start_index = slice_index * chunk_size;
        let config_cloned = config.clone();

        // fork (using std::thread instead of tokio since we are not doing async I/O here)
        threads.push(std::thread::spawn(move || {
            if config_cloned.verbose {
                //println!(
                //    "SliceID({}): ThreadID({:?}) - Trained image count: {}",
                //    slice_index,
                //    std::thread::current().id(),
                //    trained_image_slices.len()
                //);
            }

            for (_trained_image_index, train_img) in trained_image_slices.iter().enumerate() {
                let distance = match alg_cloned {
                    DistanceAlgorithm::Euclidean => {
                        // calculate per pixel distance: d(p,q) = sqrt((p-q)^2)
                        get_distance_euclidean(train_img, &test_img_cloned)
                    }
                    DistanceAlgorithm::Hamming => {
                        //
                        get_distance_hamming(train_img, &test_img_cloned)
                    }
                };
                distances_per_thread.push((
                    train_img.id, // Index of where the Trained image is
                    distance, // the distance calculated between the two images (trained vs this image)
                ));
            }
            distances_per_thread
        }));
    }

    // wait for all to forks finish/rejoin
    let mut distances: Vec<(
        TImageID,
        f64, /*distance between this image and trained image*/
    )> = Vec::new();
    for thread in threads {
        let result = thread.join().unwrap();
        distances.extend(result);
    }
    // gather/collect results from each thread
    // distances is tuple of:
    // tup.0 - TImageID
    // tup.1 - distance between the trained image and the test image
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let top_k_distances: Vec<TImageID> = distances.iter().take(config.k).map(|x| x.0).collect();
    if config.verbose {
        //println!(
        //    "\tget_k_nearest_neighbors(): ThreadID({:?}) - Trained image count: {} - found {} distances at IDs={:?}",
        //    std::thread::current().id(),
        //    trained_images.len(),
        //    config.k,
        //    top_k_distances
        //);
    }

    top_k_distances
}

//fn find_image(images: &Vec<MyImage>, id: &TImageID) -> Option<&MyImage> {
//    images.iter().find(|&&image| image.id == *id)
//}

// NOTE: neighbors should be pre-sorted, so that if the counts/encounters are the same,
// the first one will be the majority.
fn get_majority_label(
    trained_labels: &Vec<MyImage>,
    trained_neighbor_ids: &Vec<TImageID>,
) -> TClassification {
    // preserve the order of first-encounters
    let mut counts: Vec<(TClassification /*label*/, i32 /*count*/)> = Vec::new();
    for &neighbor_id in trained_neighbor_ids {
        let found_trained = trained_labels.iter().find(|&image| image.id == neighbor_id).expect("Could not find ImageID={neighbor_id}, possibly passed WRONG CHUNK of trained images!");
        if let Some((_, count)) = counts
            .iter_mut()
            .find(|(label, _)| *label == found_trained.label)
        {
            // if the label is already in the counts, increment the count
            *count += 1;
        } else {
            // push the label to the tail of the list
            counts.push((found_trained.label, 1));
        }
    }
    let mut max_count = 0;
    let mut max_label = 0;
    for (label, count) in counts {
        // first encounter should retain the label index...
        if count > max_count {
            max_count = count;
            max_label = label;
        }
    }
    max_label as TClassification
}

fn get_accuracy(trained_images: &Vec<MyImage>, test_images: &Vec<MyImage>, config: &Config) -> f64 {
    if trained_images.len() < 1 {
        panic!("ERROR: It's just moot to even bother when there are less than 1 trained images to compare against... count={}", 
            trained_images.len() );
    }
    // first, at this high level, we want to make sure to deal with the embarassingly parallel problem
    // of breaking down the test_imgs into smaller chunks and then distribute them to multiple threads
    // while these chunks are slices of the read-only trained images
    // If there are 60,000 trained images, splitting it up into 4 threads, there will be 15,000 images per thread (60,000/4=15,000)
    // But if length is 5, and thread-count is 4, then chunksize=5/4=1 element per thread, which is dumb, but for unit-test, if we can test multi-thread, it's good enough
    let min_thread_count = min(config.max_process_threads, trained_images.len());
    let chunk_size = trained_images.len() / min_thread_count;
    // pre-slice the data into chunks outside of the thread-loop so that each thread
    // does not need to clone (deep-copy) the original trained data since it can get expensive
    // memory-wise as well as performance-wise
    let mut trained_image_slices_sets: Vec<Vec<MyImage>> = Vec::new();
    for proc_index in 0..min_thread_count {
        let start = proc_index * chunk_size;
        let end = if proc_index == min_thread_count - 1 {
            // NOTE: slicing will allow end+1 even if index does not exist!
            trained_images.len()
        } else {
            min((proc_index + 1) * chunk_size, trained_images.len())
        };

        // slice (meta data) of the read-only trained data
        if start < trained_images.len() && end > start {
            trained_image_slices_sets.push(trained_images[start..end].to_vec());
        }
    }
    if config.verbose {
        println!("There are (grand total) count of {} trained images, breaking it up into {} OUTER threads (image slice blocks of {}), each thread will have {} trained images", 
            trained_images.len(), min_thread_count, trained_image_slices_sets.len(), chunk_size);
    }
    if trained_image_slices_sets.len() == 0 {
        panic!("ERROR: There are no trained images");
    }
    // tup.0 - index of the trained image
    // tup.1 - distance between the trained image and the test image
    let mut correct = 0;
    for (test_img_index, current_test_image) in test_images.iter().enumerate() {
        if config.verbose {
            //println!(
            //    "test_img={} ({} pixels)",
            //    test_img_index,
            //    current_test_image.pixels.len()
            //)
        }

        // we'll fork the threads here for each chunks of trained image slices
        let mut processes: Vec<std::thread::JoinHandle<usize>> = Vec::new();
        for (proc_index, trained_image_slices) in trained_image_slices_sets.iter().enumerate() {
            let current_test_image_cloned = current_test_image.clone();
            let config_cloned = config.clone();
            let trained_image_slice_cloned = trained_image_slices.clone(); // clone here to make sure trained_image_slices will outlive the inner loop
            processes.push(std::thread::spawn(move || {
                if config_cloned.verbose {
                    //println!(
                    //    "ProcessID({}): ThreadID({:?}) - Trained image count: {}; Image={:?}",
                    //    proc_index,
                    //    std::thread::current().id(),
                    //    trained_image_slice_cloned.len(),
                    //    current_test_image_cloned
                    //);
                }
                if trained_image_slice_cloned.len() < 1 {
                    panic!(
                        "ERROR: Found slice-block with len={}",
                        trained_image_slice_cloned.len()
                    );
                }
                let neighbors = get_k_nearest_neighbors(
                    &trained_image_slice_cloned,
                    &current_test_image_cloned,
                    &config_cloned,
                );
                if config_cloned.verbose {
                    //println!(
                    //    "ProcessID({}): ThreadID({:?}) - KN neighbors={:?}",
                    //    proc_index,
                    //    std::thread::current().id(),
                    //    neighbors
                    //)
                }
                let label = get_majority_label(&trained_image_slice_cloned, &neighbors);
                if config_cloned.verbose {
                    //println!(
                    //    "ProcessID({}): ThreadID({:?}) - Majority label={} - '{}' (Expected: {}, Matched: {})",
                    //    proc_index,
                    //    std::thread::current().id(),
                    //    label,
                    //    &CLASSES[label as TClassification],
                    //    current_test_image_cloned.label,
                    //    label == current_test_image_cloned.label,
                    //)
                    print!(
                        "{}",
                        match label == current_test_image_cloned.label {
                            true => "✓",
                            _ => "✗",
                        }
                    );
                }
                // for each partitioned trained image sets, return the label it decided to be closest
                label
            }));
        } // per outer thread

        // join them and collect what was determined, note that we do NOT care about
        // order since we cannot predict which thread completes first
        let mut labels_count_per_process: HashMap<TClassification, usize> = HashMap::new();
        for process in processes {
            let label = process.join().unwrap();
            // if exist, +1 counter, else set counter to 1
            let counter = labels_count_per_process.entry(label).or_insert(0);
            *counter += 1;
        }
        // pick the max count, note that unlike get_majority_label(),
        // the order of labels appended are undefined for we do not know
        // which process will finish first, so we'll just sort it by count
        let mut labels_sorted: Vec<(&usize /*key: label*/, &usize /*value: count*/)> =
            labels_count_per_process.iter().collect::<Vec<_>>();
        // unlike F#, sorting occurs in-place...
        // Note that because I want to pick the head (because I also need to preserve order of appearance in case of ties),
        // I am sorting in descending order by comparing rhs > lhs (as compared to lhs > rhs)
        labels_sorted.sort_by(|lhs_tup, rhs_tup| rhs_tup.1.cmp(&lhs_tup.1));
        if config.verbose {
            //let sliced_sorted = &labels_sorted[0..std::cmp::min(10, labels_sorted.len())];
            //println!("Thread rejoined returned {:?}", sliced_sorted);
        }
        // because we're sorting in descending order, picking the head (picking first() instead of last())
        let label = *labels_sorted.first().unwrap().0;

        // we're now varildating whether what was predicted matches expected
        if label == current_test_image.label {
            correct += 1;
            if config.verbose {
                //println!("\tMatched label id={}!", label)
            }
        } else if config.verbose {
            //println!(
            //    "\tIncorrect prediction, expected label {} but found {}",
            //    current_test_image.label, label
            //);
        }
    }
    // accuracy ratio:
    correct as f64 / test_images.len() as f64
}

// --data_home - path to the MNIST data
// --verbose - print debug info
// --k - number of neighbors to consider
// --algorithm - distance algorithm to use (euclidean or hamming)
// --byte_size - number of bytes to use for each pixel (1, 4, or 8)
// Default: --data_home=data/MNIST --verbose --k=3 --algorithm=euclidean --byte_size=1
fn get_config(args: &Vec<String>) -> Config {
    let mut config = Config::default();
    for arg in args {
        if arg.starts_with("--data_home=") {
            config.data_home = arg.split("=").collect::<Vec<&str>>()[1].to_string();
        } else if arg == "--verbose" {
            config.verbose = true;
        } else if arg.starts_with("--k=") {
            config.k = arg.split("=").collect::<Vec<&str>>()[1].parse().unwrap();
        } else if arg.starts_with("--algorithm=") {
            let algorithm_str = arg.split("=").collect::<Vec<&str>>()[1];
            config.algorithm = match algorithm_str {
                "euclidean" => DistanceAlgorithm::Euclidean,
                "hamming" => DistanceAlgorithm::Hamming,
                _ => panic!("ERROR: Invalid algorithm"),
            };
        } else if arg.starts_with("--byte_size=") {
            let byte_size_str = arg.split("=").collect::<Vec<&str>>()[1];
            config.byte_size = match byte_size_str {
                "1" => ByteSize::One,
                "4" => ByteSize::Four,
                "8" => ByteSize::Eight,
                _ => panic!("ERROR: Invalid byte_size"),
            };
        } else if arg.starts_with("--mp=") {
            // sets max_process_threads
            config.max_process_threads = arg.split("=").collect::<Vec<&str>>()[1].parse().unwrap();
        } else if arg.starts_with("--mt=") {
            // sets max_nearest_evaluation_threads
            config.max_nearest_evaluation_threads =
                arg.split("=").collect::<Vec<&str>>()[1].parse().unwrap();
        }
    } // for args

    config
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
    let my_trained_images = MyImage::zip(mnist.train_labels, mnist.train_imgs);
    let my_testing_images = MyImage::zip(mnist.test_labels, mnist.test_imgs);
    println!("k={}", config.k);
    println!("algorithm={:?}", config.algorithm);
    println!("byte_size={:?}", config.byte_size);

    let accuracy = get_accuracy(&my_trained_images, &my_testing_images, &config);
    println!("accuracy={}", accuracy);

    println!("elapsed time={:?}", start_time.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_dir_all;

    #[test]
    #[ignore] // NOTE: Currently, MNIST site times out causing this test to always fail
    fn test_builder() {
        let builder = MNISTBuilder::new();
        let mnist = builder
            .data_home("data/unit-test/MNIST")
            .get_data()
            .unwrap();
        assert_eq!(mnist.train_imgs.len(), 60000);
        remove_dir_all("data/unit-test/MNIST").unwrap();
    }

    #[test]
    fn test_get_distance() {
        let a = MyImage {
            id: 0,
            pixels: vec![1, 2, 3],
            label: 0,
        };
        let b = MyImage {
            id: 1,
            pixels: vec![4, 5, 6],
            label: 0,
        };
        assert_eq!(get_distance_euclidean(&a, &b), 5.196152422706632);
    }

    #[test]
    fn test_get_k_nearest_neighbors() {
        let train_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let train_labs = vec![0, 1, 2];
        let test_img = vec![1, 2, 3];

        let my_trained_images = MyImage::zip(train_labs, train_imgs);
        let my_test_image = MyImage {
            id: 0,
            label: 0,
            pixels: test_img,
        };

        assert_eq!(
            get_k_nearest_neighbors(
                &my_trained_images,
                &my_test_image,
                &Config {
                    verbose: true,
                    k: 2,
                    ..Config::default()
                }
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn test_get_majority_label() {
        let labels = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let neighbors = vec![0, 1, 2];
        let my_trained_images = MyImage::zip(labels.clone(), vec![vec![1, 2, 3]; labels.len()]);
        assert_eq!(get_majority_label(&my_trained_images, &neighbors), 1);
    }

    #[test]
    fn test_get_accuracy_single_thread() {
        // Currently (assumes that) there are 4 threads, and chunks are calculated
        // based on lengths of trained images, hence by making sure to have less than 4
        // images in trained data will assume single thread (though it'll fork that single thread)
        let train_imgs = vec![
            // NOTE: We assume there are 3 thread, and I  want to distribute each thread to have
            // same exact trained image data, so that we'll not get false-positives
            vec![1, 2, 3], // set 1
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![1, 2, 3], // set 2
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![1, 2, 3], // set 3
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];

        // Just make sure NOT to have Classification larger than MAX_CLASSIFICATION
        let train_labels = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];

        // as long as all images matches the pattern and index, we should get 100% accuracies
        let test_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let test_labels = vec![1, 2, 3];

        let my_trained_images = MyImage::zip(train_labels, train_imgs);
        let my_testing_images = MyImage::zip(test_labels, test_imgs);
        assert_eq!(
            get_accuracy(
                &my_trained_images,
                &my_testing_images,
                &Config {
                    verbose: true,
                    max_process_threads: 1,
                    max_nearest_evaluation_threads: 1,
                    ..Config::default()
                }
            ),
            // NOTE: in order to achive this 100% accuracies, based on all encounter of neighbors
            // to be tie, read the comments and logic in get_majority_label()
            1.0
        );
    }

    #[test]
    fn test_get_accuracy_multi_thread() {
        // Just make sure to have more than 4 images in trained data to test multi-threading
        let train_imgs = vec![
            // NOTE: We assume there are 3 thread, and I  want to distribute each thread to have
            // same exact trained image data, so that we'll not get false-positives
            vec![1, 2, 3], // set 1
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![1, 2, 3], // set 2
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![1, 2, 3], // set 3
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];

        // Just make sure NOT to have Classification larger than MAX_CLASSIFICATION
        let train_labels = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];

        // as long as all images matches the pattern and index, we should get 100% accuracies
        let test_imgs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let test_labels = vec![1, 2, 3];

        let my_trained_images = MyImage::zip(train_labels, train_imgs);
        let my_testing_images = MyImage::zip(test_labels, test_imgs);
        assert_eq!(
            get_accuracy(
                &my_trained_images,
                &my_testing_images,
                &Config {
                    verbose: true,
                    max_process_threads: 3, // we want 3, because we have 3 sets of trained images (sounds illogical, but to get 100% accuracies, it has to do it this way...)
                    ..Config::default()
                }
            ),
            1.0
        );
    }
}
