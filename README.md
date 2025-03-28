# vectara-eval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Twitter](https://img.shields.io/twitter/follow/vectara.svg?style=social&label=Follow%20%40Vectara)](https://twitter.com/vectara)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?style=social&logo=discord)](https://discord.com/invite/GFb8gMz6UH)

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/vectara/vectara-eval)

Vectara is the trusted GenAI platform providing simple [APIs](https://docs.vectara.com/docs/) to create conversational experiences—such as chatbots, semantic search, and question answering—from your data.

Vectara-eval is an open source python RAG evaluation toolkit used to evaluate RAG pipelines. The toolkit is modular in the sense that it is designed to be easy to add any custom metrics and evaluate the output of any RAG pipeline. 

In addition, out of the box we provide an implementation of the evaluation metrics used in TREC-RAG and also provide a connector to the Vectara RAG platform. Both of these can be used as templates to implement your own metrics and connectors to other RAG providers.

# Getting Started Guide

The evaluation framework supports integration into your RAG pipeline programatically or usage via a Flask API. This simple quick start guide however is shows you how to do an end to end evalution using our sample code. For this purpose we will be using Vectara as the platform running our RAG and the TRECRAG evaluator as our evaluator of choice. If you want to use your own RAG solution instead of Vectara skip to step 2.

## Prerequisites
* (Optional, if using Vectara as the RAG provider) A [Vectara account](https://console.vectara.com/signup) and corpus with an [API key](https://docs.vectara.com/docs/api-keys) that provides querying permissions.
* [Python 3.8 (or higher)](https://www.python.org/downloads/) with the requirements from `requirements.txt` installed in a virtual env.
* An OPENAI API Key for the LLM as a judge model.

## Step 1. Set up the Connector

This assumes you already have a Vectara account and corpus with some data indexed. Get your `customer ID` `Corpus ID` and `API Key` from the Vectara console. Update the `eval_config.yaml` file with this information. 

## Step 2. Set up the pipeline

### If using your own RAG pipeline results.
If you are joining us in step 2 with RAG outputs from your own pipeline, make sure to put your RAG output in a format that is readable by the toolkit (See `data/test_csv_connector.csv` as a sample). Update the `eval_config.yaml` file to comment out or delete the `connector` section, and instead uncomment `input_results` with the path to where your results are stored.`

### If using Vectara
 If you are following from step 1. and 2. you don't need your own RAG output, the Vectara connector will handle it for you! On the commandline, put your Vectara API key in an env variable called `VECTARA_API_KEY` (export VECTARA_API_KEY=......) the Vectara Connector will handle putting your RAG results into the right format for you. 


 In both cases, put your OpenAI API key (needed by the judge LLM) under a `OPENAI_API_KEY` env variable.

## Step 3. Set up queries to evaluate

Create a file called `queries.csv` with a column called `query`. Add the sample queries (one per row) that you want to test under this column. Here is what your CSV should look like:

| query                             |
|:---------------------------------:|
| What is a blackhole?              | 
| How big is the sun?               |
| How many moons does jupiter have? |

## Step 4. Run evaluation!

With everything configured, now is the time to run evaluation! Run the following command:

```
python run_eval.py --config eval_config.yaml
```

and you should see the evaluation progress on your command line. Once it's done, detailed results will be saved to a local CSV file where you can see the score assigned to each sample along with intermediate output useful for debugging and explainability.

## Step 5. Visualize results

You can use the `plot_results.py` script to plot results from your eval runs. Multiple different runs can be plotted on the same plot allowing for easy comparison of different configurations or RAG providers:

```
python plot_results.py metrics_1.csv metrics_2.csv 
```

# Overview

## Evaluation Workflow

The series of steps taken by the evaluation framework during an evaluation process are:

1. (Optionally) Call a RAG provider, where your data has already been indexed, with a set of queries to get the output results i.e. generated answers and retreived passages/documents.

3. Use an Evaluator to score your outputs along multiple different dimensions using one or more mertics.

4. Generate a detailed per-sample report of the scores given to the results of each query.

## Abstractions

In short **Evaluators** use one or more **Metrics** to evaluate the quality of RAG pipeline output in the form of **RAGResults**. Some evaluation metrics use judge **Models**, others may not. Evaluators produce output evaluation metrics represented as ScoredRAGResult**ScoredRAGResult** which can be written to disk as reports, for example in the CSV format.

## Metrics

Metrics are the core of the evaluation. They are used to measure the quality of the RAG system, each metric has a different focus and is used to evaluate different aspects of the RAG system. Metrics can be used to evaluate the quality of the retrieval, the quality of the (augmented) generation, the quality of the RAG system as a whole.

## Models

Models are the underlying judgement models used by some of the metrics. They are used to judge the quality of the RAG system. Models can be diverse. They may be LLMs, classifiers, rule based systems, etc.

## Evaluators

Evaluators can chain together a series of metrics to evaluate the quality of the RAG system.

## Data Classes
Several data classes are used to represent the RAG system output (which is the input for the evaluators) and the results of the evaluation.

# Web API

The framework provides a Flask-based web server that exposes endpoints for evaluation:
- `/api/v1/evaluate`: Evaluate a single RAG output
- `/api/v1/evaluate_batch`: Evaluate multiple RAG outputs

To run the server:
```
python run_server.py
```

See the [API README](/api/README.md) for detailed documentation for the API.


## Author

👤 **Vectara**

- Website: [vectara.com](https://vectara.com)
- Twitter: [@vectara](https://twitter.com/vectara)
- GitHub: [@vectara](https://github.com/vectara)
- LinkedIn: [@vectara](https://www.linkedin.com/company/vectara/)
- Discord: [@vectara](https://discord.gg/GFb8gMz6UH)

## 🤝 Contributing

Contributions, issues and feature requests are welcome and appreciated!<br />
Feel free to check [issues page](https://github.com/vectara/vectara-ingest/issues). You can also take a look at the [contributing guide](https://github.com/vectara/vectara-ingest/blob/master/CONTRIBUTING.md).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2024 [Vectara](https://github.com/vectara).<br />
This project is [Apache 2.0](https://github.com/vectara/vectara-ingest/blob/master/LICENSE) licensed.