#include <iostream>
// #include <Eigen/Dense>
#include "Eigen/Dense"
#include <math.h>
#include <unistd.h>



using Eigen::Vector4f;
using Eigen::Matrix4f;
using Eigen::MatrixXd;
using namespace std;

//4 states: [x, xdot, theta, thetadot]
Vector4f state, statedot;   //prediction of state
Matrix4f P_pre, P_post;     //prediction covariance
Matrix4f Q;                 //process covariance
Matrix4f R;                 //measurement noise covariance
Matrix4f H;                 //selects which states are observed
Vector4f h;                 //observed states

const float gravity = 9.8f;
const float masscart = 1.0f;
const float masspole = 0.1f;
const float total_mass = masspole + masscart;
const float length = 0.5f;
const float polemass_length = masspole * length;

Matrix4f I_4;

//model prediction (f)
void predict(float force, float dt){
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

//one step of the EKF using obs
void step(Vector4f obs, float force, float dt){
    //predict state and statedot using force and dt
    predict(force, dt); 
    Matrix4f F = statedot.asDiagonal();
    P_pre = F*P_post*F.transpose() + Q;

    //form observation
    H = I_4;
    h = H * obs;
    Matrix4f H_t = H.transpose();

    //calc kalman gain
    Matrix4f G = (P_pre*H_t) * ((H*P_pre*H_t).inverse() + R);

    cout << F << "\n" << endl;

    //update state
    state = state + G*(obs - h);

    //update process covariance
    P_post = (I_4 - G*H) * P_pre * (I_4 - G*H).transpose() + G*R*G.transpose(); 
}


int main(){

    float dt = 1/30.0f;
    ekf_init(0.1, 0.5, 0.25);


    while(1){

    
        float force = 0.1f;
        Vector4f obs;
        obs << 0, 0, 0.1, 0;

        step(obs, force, dt);

        sleep(1);


        cout << state << '\n' << endl;
    }
}