use flate2::bufread::MultiGzDecoder;
use ndarray::{Array1, Array2};
use std::{
    fs::File,
    io::{self, BufReader, Read},
};

pub struct MnistDataloader {
    pub directory_path: String,
    pub train_data: Array2<f64>,
    pub train_labels: Array1<u8>,
}

impl MnistDataloader {
    pub fn new(directory_path: &str) -> Self {
        MnistDataloader {
            directory_path: directory_path.to_string(),
            train_data: Array2::zeros((0, 0)),
            train_labels: Array1::zeros(0),
        }
    }

    const TRAIN_IMAGES_FILE: &str = "train-images-idx3-ubyte.gz";
    const TRAIN_LABELS_FILE: &str = "train-labels-idx1-ubyte.gz";

    pub fn load_data(&mut self) -> io::Result<()> {
        self.load_train();
        Ok(())
    }

    fn load_train(&mut self) {
        let images_path = format!("{}/{}", self.directory_path, Self::TRAIN_IMAGES_FILE);
        self.train_data = Self::read_mnist_images(&images_path).unwrap();

        let labels_path = format!("{}/{}", self.directory_path, Self::TRAIN_LABELS_FILE);
        self.train_labels = Self::read_mnist_labels(&labels_path).unwrap();
    }

    fn read_gzip(file_path: &str) -> io::Result<Vec<u8>> {
        let file = File::open(file_path)?;
        let buffered_reader = BufReader::new(file);
        let mut decoder = MultiGzDecoder::new(buffered_reader);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn read_mnist_images(file_path: &str) -> io::Result<Array2<f64>> {
        let raw_bytes = Self::read_gzip(file_path)?;

        let num_images = u32::from_be_bytes(raw_bytes[4..8].try_into().unwrap()) as usize;
        let num_rows = u32::from_be_bytes(raw_bytes[8..12].try_into().unwrap()) as usize;
        let num_cols = u32::from_be_bytes(raw_bytes[12..16].try_into().unwrap()) as usize;
        let image_size = num_rows * num_cols;

        let images_raw = &raw_bytes[16..];
        let flat_data: Vec<f64> = images_raw
            .iter()
            .map(|&byte| byte as f64 / 255.0) // Normalize each pixel
            .collect();

        let images_array = ndarray::Array::from_shape_vec((num_images, image_size), flat_data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(images_array)
    }

    fn read_mnist_labels(file_path: &str) -> io::Result<Array1<u8>> {
        let data = Self::read_gzip(file_path)?;
        let labels = ndarray::Array::from_vec(data[8..].to_vec());
        Ok(labels)
    }
}
