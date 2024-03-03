# Welcome to Bucket Brigade

_Bucket Brigade contains helper functions for using cloud storage as a simple data processing queue_

---

Cloud providers such as AWS, GCP and Azure have excellent object/blob storage capabilities. Once data is in cloud storage, it is as safe from accidental deletion or corruption as anywhere but the best self-hosted solutions.

Everything gets more complicated and more expensive once you start using the cloud providers to do more sophisticated activities such as hosting your databases and running your server instances.

This library is aimed at making it easy for developers to store their data as objects in cloud storage but use other tools for processing the data - either self-hosted or hosted in the cloud.

The key functions are:



The key function is _get_unprocessed_docs_ which lists unprocessed documents in a cloud storage bucket and sends them to a queue for serverless functions to process. The initial target for the bucketbrigade is modal.com but the intent is to accommodate any serverless function provider.

Key function:

- get_unprocessed_docs

Supporting functions:

- list_docs
- save_doc
- read_doc
- mark_completed
- skip_doc

This library has started with storage on AWS but will expand soon to GCP, Azure and others.

Thanks to https://github.com/uhjish for donating the name bucketbrigade!