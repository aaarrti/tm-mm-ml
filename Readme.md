## ML for Mater mapping

### service/ 
#### contains actual server exposing predictions over Rest + SQS listener
#### Generate server stubs with 
    cd service
    make stubs

#### Swagger UI available on http://localhost:9090/api/v1/ui

#### Run with
    cd service
    python3 main.py

### training/ 
#### contains NN fine-tuning setup as described here [MM ML Infra](https://sevensenders.atlassian.net/wiki/spaces/TRAC/pages/3220734077/ML+infra+setup)
#### Run with
    python3 training/main.py status|return_status|events|return_events

### dataset/ 
#### contains logic of extracting data from MongoDB
#### Run with 
     python3 dataset/main.py status|return_status|events|return_events|all

### shared/
#### contains shared functions, obviously

    
