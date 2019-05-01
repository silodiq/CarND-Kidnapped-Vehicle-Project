/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

std::default_random_engine gen;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

	// Initializing the number of particles
	num_particles = 50;

	// normal (Gaussian) distribution for x with std of x
	normal_distribution<double> dist_x(x, std[0]);

	// normal (Gaussian) distribution for y with std of y
	normal_distribution<double> dist_y(y, std[1]);

	// normal (Gaussian) distribution for theta with std of theta
	normal_distribution<double> angle_theta(theta, std[2]);
	
	// generate particles with normal (Gaussian) distribution means from gps.
	for (int i = 0; i < num_particles; i++) {
		
		//generate a particle
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = angle_theta(gen);		
		particle.weight = 1.0;
		
		// push particle to particles vector
		particles.push_back(particle);
	}

	// indicate particle filter has been initialized
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	/**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */


	//add noise to velocity and yaw rate., seems like vector:particles can't multiply directly with motion model.
	// the standard variance although extract from sigma_pos from ParticleFilter::init, the each function inclass is saperated , so 
	// if we want to use the std_sigma of each parameters, we should re-extract again.

	
	// normal (Gaussian) distribution for x ,mean in 0
	normal_distribution<double> dist_x(0, std_pos[0]);

	// normal (Gaussian) distribution for y ,mean in 0
	normal_distribution<double> dist_y(0, std_pos[1]);	

	// normal (Gaussian) distribution for theta ,mean in 0
	normal_distribution<double> angle_theta(0, std_pos[2]);
	
	// predict every particle's motion
	for (int i = 0; i < num_particles; i++) {
		
		// yaw_rate is none zero
		if (fabs(yaw_rate) >= EPS) {
			particles[i].x = particles[i].x + (velocity / yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y = particles[i].y + (velocity / yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		}
		// we think theat is same with last step
		else {
			particles[i].x = particles[i].x + velocity * delta_t *cos(particles[i].theta);
			particles[i].y = particles[i].y + velocity * delta_t *sin(particles[i].theta);
		}
		
		// add noise to particle
		particles[i].x = particles[i].x + dist_x(gen);
		particles[i].y = particles[i].y + dist_y(gen);
		particles[i].theta = particles[i].theta + angle_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	/**
	 * TODO: Find the predicted measurement that is closest to each
	 *   observed measurement and assign the observed measurement to this
	 *   particular landmark.
	 * NOTE: this method will NOT be called by the grading code. But you will
	 *   probably find it useful to implement this method and use it as a helper
	 *   during the updateWeights phase.
	 */

	int n_observations = observations.size();
	int n_predictions = predicted.size();

	for (int i = 0; i < n_observations; i++) {
		
		// initialize min distance as a max number
		double min_distance = numeric_limits<double>::max();

		// initialize id ,not in the map
		int id_in_predicted = -1;
		
		for (int j = 0; j < n_predictions; j++) {
			//use helper's function
			// calculate distance between observation and predicted
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			// update mini distance and id 
			if (distance < min_distance) {
				min_distance = distance;
				id_in_predicted = predicted[j].id;
			}
		}
		
		// means, this observation is belong to the landmark in predicted
		observations[i].id = id_in_predicted;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	/**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

	// std of landmark in x, y 
	double std_landmark_range = std_landmark[0];
	double std_landmark_bearing = std_landmark[1];

	// in order to find landmarks in sensor's detection range
	double sensor_range_2 = sensor_range * sensor_range;

	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		// used to save landmarks in sensor's detection range
		vector<LandmarkObs> in_range_landmarks;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int id = map_landmarks.landmark_list[j].id_i;
			double delta_x = x - landmark_x;
			double delta_y = y - landmark_y;
			
			// get landmarks in sensor's detection range
			if (delta_x*delta_x + delta_y * delta_y <= sensor_range_2) {
				in_range_landmarks.push_back(LandmarkObs{ id, landmark_x, landmark_y });
			}
		}

		
		// transform coordinate from car's to map's for ovservation
		vector<LandmarkObs> transform_observations;
		
		for (int j = 0; j < observations.size(); j++) {

			double transformed_x = x + cos(theta)*observations[j].x - sin(theta) * observations[j].y;
			double transformed_y = y + sin(theta)*observations[j].x + cos(theta) * observations[j].y;
			
			transform_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}
		
		// association of landmark and observation
		dataAssociation(in_range_landmarks, transform_observations);

		particles[i].weight = 1.0;

		//weights calculation
		for (int j = 0; j < transform_observations.size(); j++) {
			double observation_x = transform_observations[j].x;
			double observation_y = transform_observations[j].y;

			int landmark_id = transform_observations[j].id;

			double landmark_x;
			double landmark_y;

			int k = 0;
			int num_landmarks = in_range_landmarks.size();
			bool found = false;
			while (!found && k < num_landmarks) {
				if (in_range_landmarks[k].id == landmark_id) {
					found = true;
					landmark_x = in_range_landmarks[k].x;
					landmark_y = in_range_landmarks[k].y;
				}
				k++;
			}
			
			double delta_x = observation_x - landmark_x;
			double delta_y = observation_y - landmark_y;
			
			// assume correction of x and y direction is independent
			//update weight
			double weight = (1 / (2 * M_PI*std_landmark_range*std_landmark_bearing)) * exp(-(delta_x*delta_x / (2 * std_landmark_range*std_landmark_range) + (delta_y*delta_y / (2 * std_landmark_bearing*std_landmark_bearing))));

			if (0 == weight) {
				particles[i].weight = particles[i].weight*EPS;
			}
			else {
				particles[i].weight = particles[i].weight * weight;
			}
		}
	}
}

void ParticleFilter::resample() {
	/**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    // get weiths and max weight
	vector<double> weights;
	double max_weight = numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > max_weight) {
			max_weight = particles[i].weight;
		}
	}

	uniform_real_distribution<float> dist_float(0.0, max_weight);
	uniform_int_distribution<int> dist_int(0, num_particles - 1);

	int index = dist_int(gen);

	double beta = 0.0;

	// sampling wheel
	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++) {
		beta += dist_float(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to which assign each listed association, 
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
	vector<double> v;

	if (coord == "X") {
		v = best.sense_x;
	}
	else {
		v = best.sense_y;
	}

	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
