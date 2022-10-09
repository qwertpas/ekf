#include <iostream>
// #include <Eigen/Dense>
#include "Eigen/Dense"
#include <math.h>

using Eigen::Vector4f;
using Eigen::Matrix4f;
using Eigen::MatrixXd;

//4 states: [x, xdot, theta, thetadot]
Vector4f state, statedot;
Matrix4f P_pre, P_post;     //prediction covariance
Matrix4f Q;                 //process covariance
Matrix4f R;                 //measurement noise covariance
Vector4f obs;               //observations
Matrix4f H;                 //selects which states are observed
Vector4f h;                 //observed states

const float gravity = 9.8f;
const float masscart = 1.0f;
const float masspole = 0.1f;
const float total_mass = masspole + masscart;
const float length = 0.5f;
const float polemass_length = masspole * length;

Matrix4f I_4;

float dt = 1/30.0f;
float force = 0.0f;

//model prediction (f)
void predict(){
    float x = state(0);
    float xdot = state(1);
    float theta = state(2);
    float thetadot = state(3);

    float sintheta = sin(theta);
    float costheta = cos(theta);

    float temp = (force + polemass_length * thetadot*thetadot * sintheta) / total_mass;
    float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta*costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    //Euler integration
    x += dt * xdot;
    xdot += dt * xacc;
    theta += dt * thetadot;
    thetadot += dt * thetaacc;

    state(0) = x;
    state(1) = xdot;
    state(2) = theta;
    state(3) = thetadot;

    statedot(0) = xdot;
    statedot(1) = xacc;
    statedot(2) = thetadot;
    statedot(3) = thetaacc;
}

//select which states are being observed
void form_obs(){
    H = I_4; //observe all 4 states
    h = H * obs;
}

void ekf_init(float pval, float qval, float rval){
    I_4 <<  1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    state << 0, 0, 0, 0;
    P_post = I_4 * pval;
    Q = I_4 * qval;
    R = I_4 * rval;
}

//one step of the EKF using the observation obs
void step(){
    predict(); //updates state and statedot using force and dt
    P_pre = statedot * P_post * statedot + Q;

    form_obs(); //updates h and H, the formatted observations
    Matrix4f H_t = H.transpose();

    Matrix4f G = (P_pre*H_t) * ((H*P_pre*H_t).inverse() + R);

    state = state + G*(obs - h);



}


int main(){


    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 3.42;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;
}