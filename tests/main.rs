use ndarray::Array;
use ndarray::{Array1, ArrayView1, Ix2};

//mod lib;
use histnd::{binary_search, histnd_parallel, histnd_serial, Weight};

#[test]
fn check_binary_search() {
    let testitem = ArrayView1::from(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]);

    let ret1 = binary_search(&testitem, &0.1);
    let ret2 = binary_search(&testitem, &10.);
    let ret3 = binary_search(&testitem, &5.2);
    //println!("{:?}, {:?}, {:?}", ret1, ret2, ret3);
    assert_eq!(ret1, Err(1));
    assert_eq!(ret2, Ok(10));
    assert_eq!(ret3, Err(6));
}

#[test]
fn check_parallel_and_serial() {
    // init bin
    //let bin = ArrayView1::from(&[-1,0,1,2,3,4,5,7,8,9,10,100]);
    let bin: Array1<f64> = Array::range(0., 500., 1.5);
    let testitem_nd = [bin.view(), bin.view(), bin.view()];
    // init samples
    let testsample: Array1<f64> = Array::range(0., 300000., 1.);
    let testsample_nd = testsample.view().into_shape(Ix2(100000, 3)).unwrap();
    // init weights
    let raw_weight = Array::range(0., 5., 5. / 100000.);

    println!("input data: \n{:?}\n\n", testsample_nd);
    println!("bins: \n{:?}\n\n", testitem_nd);

    let my_weight = Weight::HasWeight(raw_weight.view());
    let histndret = histnd_serial(testsample_nd, &testitem_nd, my_weight);

    let my_weight = Weight::HasWeight(raw_weight.view());
    let histndret_p = histnd_parallel(testsample_nd, &testitem_nd, my_weight, 10);
    println!("histnd_p returns: \n{:?}\n\n", histndret_p);

    assert_eq!(histndret, histndret_p);
}
