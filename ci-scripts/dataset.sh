#!/bin/sh

aws eks update-kubeconfig --name track-dev

# clean up if there is a job left from previous unfinished run

kubectl delete job/mm-ml-dataset --namespace=datahub-dev || true

# create job and wait for it to start
kubectl apply -f dataset/k8s/job.yaml --namespace=datahub-dev

kubectl wait --for=condition=ready pod -l app=mm-ml-dataset --timeout=5m --namespace=datahub-dev

# stream logs
kubectl logs -l job-name=mm-ml-dataset --follow --namespace=datahub-dev

# clean up afterwards
kubectl delete job/mm-ml-dataset --namespace=datahub-dev
