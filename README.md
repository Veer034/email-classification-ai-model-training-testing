# email-classification-ai

### Install:
    pip install --user google-cloud-storage



### To Login:
    gcloud auth application-default login



### Connect the ssh to instance:

    gcloud compute ssh --zone "asia-south1-a" "instance-20240912-185648" --project "convonest-dev"

Enter passphrase (empty for no passphrase):  convonest
Enter same passphrase again: convonest
Your identification has been saved in /Users/ranveersingh/.ssh/google_compute_engine
Your public key has been saved in /Users/ranveersingh/.ssh/google_compute_engine.pub
The key fingerprint is:
SHA256:/gE9T4CVw9G1oNPRCsIsoLiOk9bBbsNrqlCkuAAe7l0 ranveersingh@ZL-BLR-MAC140.local
The key's randomart image is:
+---[RSA 3072]----+
|    .. o .o+oo.  |
| . .  . +o=o.o.. |
|o.o    ...+o...  |
|=+o      . o.    |
|=o.o E  S o .    |
|=+= o  . . +     |
|*+ B    . . .    |
|o...o    . .     |
|o.o.      .      |
+----[SHA256]-----+



### Commands to install library prior to running roBERTa model:

    sudo apt update
    sudo apt install python3-pip
    pip3 install pandas torch transformers evaluate datasets google-cloud-storage sentencepiece



### If error comes related to datasets version
    pip install --upgrade datasets


### Update Dependencies:
    pip install accelerate -U

    pip install transformers[torch] -U


### Create python file:

    vim train_roberta.py


### Clear VM Transformers Cache:

    sudo rm -rf ~/.cache/huggingface/transformers

### Class All Caches:
    sudo apt clean && sudo apt autoremove



### Run Python code:

    nohup python3 spam_ham_checker.py > output.log 2>&1 &


### Copy Model to Buckets:

gsutil cp * gs://email_classification_convonest/final_spam_ham_model/DeBERTa/4_classification


Copy model from bucket to local:

gsutil cp -r gs://email_classification_convonest/final_spam_ham_model/RoBERTa/filtered_spam_ham .


gsutil cp -r gs://email_classification_convonest/final_spam_ham_model/DeBERTa/4_classification .


Check the log file:

tail -f output.log


Check if the process is still running:
ps aux | grep train_roberta.py

Monitor system resources:
top

