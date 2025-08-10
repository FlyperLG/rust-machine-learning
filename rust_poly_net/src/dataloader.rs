// inspired by the mnist_reader crate located at: https://docs.rs/mnist_reader/latest/mnist_reader/
use crate::number_representations::core::MlScalar;
use flate2::bufread::MultiGzDecoder;
use ndarray::{Array1, Array2};
use std::{
    cmp,
    fs::File,
    io::{self, BufReader, Read},
};

pub struct MnistDataloader<T> {
    pub directory_path: String,
    pub train_data: Array2<T>,
    pub train_labels: Array1<u8>,
    pub test_data: Array2<T>,
    pub test_labels: Array1<u8>,
    pub train_limit: Option<usize>,
    pub test_limit: Option<usize>,
}

impl<T: MlScalar> MnistDataloader<T> {
    pub fn new(
        directory_path: &str,
        train_limit: Option<usize>,
        test_limit: Option<usize>,
    ) -> Self {
        MnistDataloader {
            directory_path: directory_path.to_string(),
            train_data: Array2::zeros((0, 0)),
            train_labels: Array1::zeros(0),
            test_data: Array2::zeros((0, 0)),
            test_labels: Array1::zeros(0),
            train_limit: train_limit,
            test_limit: test_limit,
        }
    }

    const TRAIN_IMAGES_FILE: &str = "train-images-idx3-ubyte.gz";
    const TRAIN_LABELS_FILE: &str = "train-labels-idx1-ubyte.gz";
    const TEST_IMAGES_FILE: &str = "t10k-images-idx3-ubyte.gz";
    const TEST_LABELS_FILE: &str = "t10k-labels-idx1-ubyte.gz";

    pub fn load_data(&mut self) -> io::Result<()> {
        self.load_train();
        self.load_test();
        Ok(())
    }

    fn load_train(&mut self) {
        let images_path = format!("{}/{}", self.directory_path, Self::TRAIN_IMAGES_FILE);
        self.train_data = Self::read_mnist_images(&images_path, self.train_limit).unwrap();

        let labels_path = format!("{}/{}", self.directory_path, Self::TRAIN_LABELS_FILE);
        self.train_labels = Self::read_mnist_labels(&labels_path, self.train_limit).unwrap();
    }

    fn load_test(&mut self) {
        let images_path = format!("{}/{}", self.directory_path, Self::TEST_IMAGES_FILE);
        self.test_data = Self::read_mnist_images(&images_path, self.test_limit).unwrap();

        let labels_path = format!("{}/{}", self.directory_path, Self::TEST_LABELS_FILE);
        self.test_labels = Self::read_mnist_labels(&labels_path, self.test_limit).unwrap();
    }

    fn read_gzip(file_path: &str) -> io::Result<Vec<u8>> {
        let file = File::open(file_path)?;
        let buffered_reader = BufReader::new(file);
        let mut decoder = MultiGzDecoder::new(buffered_reader);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn read_mnist_images(file_path: &str, limit: Option<usize>) -> io::Result<Array2<T>> {
        let raw_bytes = Self::read_gzip(file_path)?;

        let total_images = u32::from_be_bytes(raw_bytes[4..8].try_into().unwrap()) as usize;
        let num_rows = u32::from_be_bytes(raw_bytes[8..12].try_into().unwrap()) as usize;
        let num_cols = u32::from_be_bytes(raw_bytes[12..16].try_into().unwrap()) as usize;
        let image_size = num_rows * num_cols;

        let num_images_to_load = match limit {
            Some(lim) => cmp::min(lim, total_images),
            None => total_images,
        };

        let images_data_end = 16 + num_images_to_load * image_size;
        let images_raw = &raw_bytes[16..images_data_end];
        let flat_data: Vec<T> = images_raw
            .iter()
            .map(|&byte| T::from(byte as f64).unwrap() / T::from(255.0).unwrap()) // Normalize each pixel
            .collect();

        let images_array =
            ndarray::Array::from_shape_vec((num_images_to_load, image_size), flat_data)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(images_array)
    }

    fn read_mnist_labels(file_path: &str, limit: Option<usize>) -> io::Result<Array1<u8>> {
        let data = Self::read_gzip(file_path)?;
        let total_labels = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;

        let num_labels_to_load = match limit {
            Some(lim) => cmp::min(lim, total_labels),
            None => total_labels,
        };

        let labels_data_end = 8 + num_labels_to_load;
        let labels_slice = &data[8..labels_data_end];

        let labels = ndarray::Array::from_vec(labels_slice.to_vec());
        Ok(labels)
    }
}
