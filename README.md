﻿# MNIST-model
This is a small project that i created for my internship.
Using Python, Docker and Kubernetes i trained an MNIST model using Deep learning.
I created an API endpoint to use this trained model.
After that i put this API inside of a Docker Container and ran it in Kubernetes.

## Minikube
To set up you don't need the Dockerfile (It's included for academic purposes).
Follow these steps if you want to run the project using Minikube.
Just open a terminal inside the k8s file and do the following

`kubectl apply -f mnist-deployment.yaml`

and

`kubectl apply -f mnist-service.yaml`

After this the containers will start creating, this can take a while (especially if you have slow internet)
