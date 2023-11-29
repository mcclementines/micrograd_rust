//! tests/simple_model_test.rs

use micrograd_rust::{mlp::Mlp, value::Value};

#[test]
fn test_manual_training_loop() {
    // Seems like the most impact on this model is
    // to change the number of neurons in the initial layer,
    // the more neurons, the better the network is trained.
    //
    // More layers and more neurons in additional layers
    // does not impact results much or even makes the training
    // worse
    //
    // actually, this result might have been from not calling
    // zero_grad on the network before calling backward again
    let mlp = Mlp::new(3, vec![4,4,1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let ys = [1.0, -1.0, -1.0, 1.0];

    let mut loss: Value = Value::new(10.0);
    
    for _ in 0..30 {
        // Forward pass
        let ypred: Vec<Value> = xs.iter().map(|input| mlp.callf(input).get(0).unwrap().to_owned()).collect();
        loss = ypred.iter().zip(ys).map(|(pred, ygt)| (pred-ygt).powf(2.0)).sum();
        
        // backward pass
        mlp.parameters().iter().for_each(|p| {
            p.set_grad(0.0);
        });
        loss.backward();
        
        // update
        mlp.parameters().iter().for_each(|p| {
            p.set_data(p.data() + -0.06 * p.grad());
        });
        
        ypred.iter().for_each(|pred| println!("data: {:?}, loss: {:?}", pred.data(), loss.data()));

        if loss.data() < 0.06 {
            break;
        }
    }
    
    // may fail due to random chance
    // which means it is not a good test
    // but oh well
    assert!(loss.data() < 0.06);
}
