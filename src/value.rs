//! src/value.rs

use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops;
use std::rc::Rc;

struct InnerValue {
    data: f32,
    grad: f32,
    backward: Box<dyn Fn()>,
    prev: Vec<Value>,
    op: String,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<InnerValue>>);

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={:?}, grad={:?}, prev={:?}, op={:?})",
            self.data(),
            self.grad(),
            self.prev(),
            self.op()
        )
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data())
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl PartialEq<f32> for Value {
    fn eq(&self, other: &f32) -> bool {
        self.data() == *other
    }
}

impl PartialEq<Value> for f32 {
    fn eq(&self, other: &Value) -> bool {
        *self == other.data()
    }
}

impl Value {
    pub fn new(data: f32) -> Value {
        let grad: f32 = 0.0;
        let backward = Box::new(|| {});
        let prev = Vec::<Value>::new();
        let op = String::new();

        let value = InnerValue {
            data,
            grad,
            backward,
            prev,
            op,
        };

        Value(Rc::new(RefCell::new(value)))
    }

    pub fn with_op(data: f32, children: Option<Vec<Value>>, op: &str) -> Value {
        let grad: f32 = 0.0;
        let backward = Box::new(|| {});
        let prev = match children {
            Some(c) => c,
            None => Vec::<Value>::new(),
        };
        let op = String::from(op);

        let value = InnerValue {
            data,
            grad,
            backward,
            prev,
            op,
        };

        Value(Rc::new(RefCell::new(value)))
    }

    pub fn powf(&self, pow: f32) -> Value {
        self.powv(&Value::new(pow))
    }

    pub fn powv(&self, pow: &Value) -> Value {
        let data = self.data().powf(pow.data());
        let children = vec![self.to_owned(), pow.to_owned()];

        let value = Value::with_op(data, Some(children), "pow");

        let v = value.clone();
        let s = self.clone();
        let p = pow.clone();
        value.set_backward(move || {
            s.accumulate_grad((p.data() * s.data().powf(p.data() - 1.0)) * v.grad());
        });

        value
    }

    pub fn exp(&self) -> Value {
        let data = self.data().exp();
        let children = vec![self.to_owned()];

        let value = Value::with_op(data, Some(children), "exp");

        let v = value.clone();
        let s = self.clone();
        value.set_backward(move || {
            s.accumulate_grad(v.data() * v.grad());
        });

        value
    }

    pub fn tanh(&self) -> Value {
        // just flexing
        let ref num = &(2.0 * self).exp() - 1.0;
        let ref den = &(2.0 * self).exp() + 1.0;
        let value = num / den;
        value.set_op("tanh");

        let v = value.clone();
        let s = self.clone();
        value.set_backward(move || {
            s.accumulate_grad((1.0 - v.data().powi(2)) * v.grad());
        });

        value
    }

    pub fn is_in(&self, values: &Vec<Value>) -> bool {
        for value in values {
            if Rc::ptr_eq(&self.0, &value.0) {
                return true;
            }
        }

        false
    }

    pub fn build_topo(&self, visited: &mut Vec<Value>, topo: &mut Vec<Value>) -> Vec<Value> {
        if !self.is_in(&visited) {
            visited.push(self.clone());

            for child in self.prev() {
                *topo = child.build_topo(visited, topo);
            }

            topo.push(self.clone());
        }

        topo.clone()
    }

    pub fn backward(&self) {
        let mut topo = self.build_topo(&mut Vec::<Value>::new(), &mut Vec::<Value>::new());
        topo.reverse();

        self.set_grad(1.0);
        topo.iter().for_each(|value| value.once_backward());
    }

    pub fn once_backward(&self) {
        (self.0.borrow().backward)();
    }

    pub fn set_backward<F: Fn() + 'static>(&self, backward: F) {
        self.0.borrow_mut().backward = Box::new(backward);
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data.clone()
    }

    pub fn set_data(&self, data: f32) {
        self.0.borrow_mut().data = data;
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad.clone()
    }

    pub fn set_grad(&self, grad: f32) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn accumulate_grad(&self, grad: f32) {
        self.0.borrow_mut().grad += grad;
    }

    pub fn prev(&self) -> Vec<Value> {
        self.0.borrow().prev.clone()
    }

    pub fn set_prev(&self, prev: Vec<Value>) {
        self.0.borrow_mut().prev = prev;
    }

    pub fn op(&self) -> String {
        self.0.borrow().op.clone()
    }

    pub fn set_op(&self, op: &str) {
        self.0.borrow_mut().op = String::from(op);
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Value {
        iter.fold(Value::new(0.0), ops::Add::add)
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        &self + &rhs
    }
}

impl ops::Add for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Value {
        let result = self.data() + rhs.data();
        let children = vec![self.to_owned(), rhs.to_owned()];

        let value = Value::with_op(result, Some(children), "+");

        let v = value.clone();
        let l = self.clone();
        let r = rhs.clone();
        value.set_backward(move || {
            l.accumulate_grad(1.0 * v.grad());
            r.accumulate_grad(1.0 * v.grad());
        });

        value
    }
}

impl ops::Add<&Value> for f32 {
    type Output = Value;

    fn add(self, rhs: &Value) -> Value {
        let lhs = &Value::new(self);

        lhs + rhs
    }
}

impl ops::Add<f32> for &Value {
    type Output = Value;

    fn add(self, rhs: f32) -> Value {
        let rhs = &Value::new(rhs);

        self + rhs
    }
}

impl ops::Sub for &Value {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Value {
        let value = self + &(-rhs);
        value.set_op("-");

        value
    }
}

impl ops::Sub<&Value> for f32 {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Value {
        let value = self + &(-rhs);
        value.set_op("-");

        value
    }
}

impl ops::Sub<f32> for &Value {
    type Output = Value;

    fn sub(self, rhs: f32) -> Value {
        let value = self + (-rhs);
        value.set_op("-");

        value
    }
}

impl ops::Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Value {
        let result = self.data() * rhs.data();
        let children = vec![self.to_owned(), rhs.to_owned()];

        let value = Value::with_op(result, Some(children), "*");

        let v = value.clone();
        let l = self.clone();
        let r = rhs.clone();
        value.set_backward(Box::new(move || {
            l.accumulate_grad(r.data() * v.grad());
            r.accumulate_grad(l.data() * v.grad());
        }));

        value
    }
}

impl ops::Mul<f32> for &Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Value {
        let rhs = &Value::new(rhs);

        self * rhs
    }
}

impl ops::Mul<&Value> for f32 {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Value {
        let lhs = &Value::new(self);

        lhs * rhs
    }
}

impl ops::Div for &Value {
    type Output = Value;

    fn div(self, rhs: &Value) -> Value {
        let value = self * &rhs.powf(-1.0);
        value.set_op("/");

        value
    }
}

impl ops::Div<f32> for &Value {
    type Output = Value;

    fn div(self, rhs: f32) -> Value {
        let value = self * rhs.powf(-1.0);
        value.set_op("/");

        value
    }
}

impl ops::Div<&Value> for f32 {
    type Output = Value;

    fn div(self, rhs: &Value) -> Value {
        let value = self * &rhs.powf(-1.0);
        value.set_op("/");

        value
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        &self * -1.0
    }
}

impl ops::Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn test_value() {
        let v = Value::new(5.0);
        let f: f32 = 5.0;

        assert_eq!(
            v.data(),
            f,
            "testing Value initialization with float {:?}",
            f
        );
    }

    #[test]
    fn test_value_add() {
        let a = &Value::new(5.0);
        let b = &Value::new(2.2);
        let result = a + b;
        let expected: f32 = 7.2;

        assert_eq!(
            result.data(),
            expected,
            "testing Value add with {:?} and {:?}",
            a,
            b
        );
        assert_eq!(result.op(), String::from("+"));
        assert_eq!(
            result.prev().len(),
            2,
            "testing if new Value has 2 children"
        );
        println!("add result: {:?}", result);
    }

    #[test]
    fn test_value_add_backward() {
        let a = &Value::new(5.0);
        let b = &Value::new(2.2);
        let result = a + b;
        result.set_grad(5.0);

        result.once_backward();

        assert_eq!(a.grad(), 5.0);
        assert_eq!(b.grad(), 5.0);
    }

    #[test]
    fn test_value_mul() {
        let a = &Value::new(5.0);
        let b = &Value::new(2.0);
        let result = a * b;
        let expected: f32 = 10.0;

        assert_eq!(
            result.data(),
            expected,
            "testing Value mul with {:?} and {:?}",
            a,
            b
        );
        assert_eq!(result.op(), String::from("*"));
        assert_eq!(
            result.prev().len(),
            2,
            "testing if new Value has 2 children"
        );
        println!("mul result: {:?}", result);
    }

    #[test]
    fn test_value_mul_backward() {
        let a = &Value::new(5.0);

        let result = a * 2.0;
        result.set_grad(2.0);
        result.once_backward();

        assert_eq!(a.grad(), 4.0);
    }

    #[test]
    fn test_value_tanh() {
        let a = &Value::new(2.0);
        let result = a.tanh();
        let expected: f32 = (2.0_f32).tanh();

        assert_eq!(result.data(), expected, "testing Value tanh on {:?}", a);
    }

    #[test]
    fn test_value_tanh_backward() {
        let a = &Value::new(2.0);

        let result = a.tanh();
        result.set_grad(2.0);
        result.once_backward();

        let expected: f32 = (1.0 - a.data().tanh().powi(2)) * result.grad();

        assert_eq!(a.grad(), expected, "testing Value tanh backward on {:?}", a);
    }

    #[test]
    fn test_value_backward() {
        let x1 = &Value::new(2.0);
        let x2 = &Value::new(0.0);

        let w1 = &Value::new(-3.0);
        let w2 = &Value::new(1.0);

        let b = &Value::new(6.8813735870195432);

        let ref x1w1 = x1 * w1;
        let ref x2w2 = x2 * w2;
        let ref x1w1x2w2 = x1w1 + x2w2;
        let ref x1w1x2w2b = x1w1x2w2 + b;

        let ref out = x1w1x2w2b.tanh();
        out.set_grad(1.0);
        out.backward();

        assert_eq!(out.data(), 0.7071067);
        assert_eq!(x1w1x2w2b.grad(), 0.5000001);
        assert_eq!(x1.grad(), -1.5000004);
    }

    #[test]
    fn test_value_grad_accumulates() {
        let a = &Value::new(2.0);
        let ref result = a + a;
        result.backward();

        assert_eq!(a.grad(), 2.0);

        let a = &Value::new(2.0);
        let ref result = a * a;
        result.backward();

        assert_eq!(a.grad(), 4.0);
    }
}
