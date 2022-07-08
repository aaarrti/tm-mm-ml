#!/bin/sh

TASK=$1

echo "executing task $TASK"

aws eks update-kubeconfig --name track-dev

# clean up if there is a job left from previous unfinished run
kubectl delete job/mm-ml-train-"$TASK" --namespace=datahub-dev || true

kubectl apply -f training/k8s/"$TASK"-job.yaml

# wait for spot.io to spin up a GPU node
# kubectl wait --for=condition=ready node -l nodegroup=gpu_support --timeout=15m --namespace=datahub-dev

kubectl wait --for=condition=ready pod -l app=mm-ml-train-"$TASK" --timeout=15m --namespace=datahub-dev

kubectl logs -l job-name=mm-ml-train-"$TASK" --follow --namespace=datahub-dev

#clean up
kubectl delete job/mm-ml-train-"$TASK" --namespace=datahub-dev
