//transform = transforms.Compose([
    //transforms.Resize((224,224)), # resize the images to 224x24 pixels
    //transforms.ToTensor(), # convert the images to a PyTorch tensor
    //transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
//])
//The first aim is to run the images through a ResNet50 model in
//parallel
//http://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/
//https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md

//use tch::vision::resnet;
use tch::nn::{self, FuncT, ModuleT, Conv2D};
//use tch::vision;

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, padding, bias: false, ..Default::default() };
    nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        nn::seq_t().add(conv2d(&p / "0", c_in, c_out, 1, 0, stride)).add(nn::batch_norm2d(
            &p / "1",
            c_out,
            Default::default(),
        ))
    } else {
        nn::seq_t()
    }
}


fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv1).apply_t(&bn1, train).relu().apply(&conv2).apply_t(&bn2, train);
        (xs.apply_t(&downsample, train) + ys).relu()
    })
}

fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
    let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn resnet(
    p: &nn::Path,
    nclasses: Option<i64>,
    c1: i64,
    c2: i64,
    c3: i64,
    c4: i64,
) -> FuncT<'static> {
    let conv1 = conv2d(p / "conv1", 3, 64, 7, 3, 2);
    let bn1 = nn::batch_norm2d(p / "bn1", 64, Default::default());
    let layer1 = basic_layer(p / "layer1", 64, 64, 1, c1);
    let layer2 = basic_layer(p / "layer2", 64, 128, 2, c2);
    let layer3 = basic_layer(p / "layer3", 128, 256, 2, c3);
    let layer4 = basic_layer(p / "layer4", 256, 512, 2, c4);
    let fc = nclasses.map(|n| nn::linear(p / "fc", 512, n, Default::default()));
    nn::func_t(move |xs, train| {
        xs.apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&layer1, train)
            //.apply_t(&layer2, train)
            //.apply_t(&layer3, train)
            //.apply_t(&layer4, train)
            //.adaptive_avg_pool2d(&[1, 1])
            //.flat_view()
            //.apply_opt(&fc)
    })
}
fn main() {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet(&vs.root(),None, 2, 2, 2, 2);
    println!("{:?}", &vs.root());
    vs.load("/home/maria/Documents/Weights/resnet18.ot");
    
}