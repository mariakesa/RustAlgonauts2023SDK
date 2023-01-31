//transform = transforms.Compose([
    //transforms.Resize((224,224)), # resize the images to 224x24 pixels
    //transforms.ToTensor(), # convert the images to a PyTorch tensor
    //transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
//])
//The first aim is to run the images through a ResNet50 model in
//parallel
//http://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/
//https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md

use tch::vision::resnet;
use tch::nn;
fn main() {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet::resnet18_no_final_layer(&vs.root());
    println!("{:?}", vs.root());
    vs.load("/home/maria/Documents/Weights/resnet18.ot");
}