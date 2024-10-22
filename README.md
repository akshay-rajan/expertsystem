# EXPERT SYSTEM


Machine Learning is a vast and complex field that requires a lot of dedication and time to master. 
However, the complexity of ML algorithms and the necessity for programming skills create
significant barriers for individuals who wish to leverage ML for their projects or research. 
But what if we could remove this constraint and allow the users to build models without any coding knowledge?

**Expert System** is a platform that allows users to build, train, and test Machine Learning Models without the need for programming languages. 

It provides a user-friendly interface where users can:

1. Upload the data
2. Clean and process it
3. Select an algorithm
4. Train the model
5. Perform evaluations and predictions

in just a few clicks.

Expert System is designed to make Machine Learning accessible to everyone, regardless of their coding knowledge. 

With Expert System, Machine Learning is no longer limited to programmers and data scientists. 
It is open to everyone who wants to harness the power of data and build intelligent systems.

### Design

```mermaid
flowchart LR
    A(Start) --> B[Select Algorithm]
    B --> C1[Classification]
    B --> C2[Regression]
    B --> C3[Clustering]

    C1 --> D0[KNN]
    C1 --> D1[Logistic Regression]
    C1 --> D2[Naive Bayes]
    C1 --> D3[SVM]
    C1 --> D4[Decision Tree]
    C1 --> D5[Random Forest]

    C2 --> D6[Linear Regression]
    C2 --> D7[Lasso Regression]
    C2 --> D8[Ridge Regression]
    C2 --> D9[Decision Tree]
    C2 --> D10[Random Forest]

    C3 --> D11[K Means]
    C3 --> D12[Hierarchical]

    D0 --> E(Upload Data)
    D1 --> E(Upload Data)
    D2 --> E(Upload Data)
    D3 --> E(Upload Data)
    D4 --> E(Upload Data)
    D5 --> E(Upload Data)
    D6 --> E(Upload Data)
    D7 --> E(Upload Data)
    D8 --> E(Upload Data)
    D9 --> E(Upload Data)
    D10 --> E(Upload Data)
    D11 --> E(Upload Data)
    D12 --> E(Upload Data)

    E --> F(Feature Selection)
    F --> G(Hyperparameter Selection)
    G --> H(Modelling)
    H --> I(Evaluation)
    H --> J(Prediction)
    I --> K(End)
    J --> K(End)
```


### *Technologies*

1. *Django*
2. *d3.js*
3. *Scikit-learn*
4. *Matplotlib*, *Pandas*, *Numpy* etc.

### CI/CD Pipeline

Github Actions is used to automate the developer workflow:
**Development**, **Testing**, **Build** and **Deployment**.

![alt](./others/Workflow.png)

[django.yml](.github/workflows/django.yml)

<!-- > Github Actions **Listen** to Github **Events**, such as a PR, Contributor addition etc. -->
<!-- > The Event **Triggers a Workflow**, which contain **Actions**, for example Sorting, Labelling, Assignment to someone etc. -->
<!-- [django.yml](.github/workflows/django.yml) -->

### *Usage*

Clone the project:
```
git clone https://github.com/akshay-rajan/expertsystem.git
```
Navigate to the project directory:
```
cd expertsystem
```
Create a virutal environment:
```bash
python -m venv myenv
```
Activate the virtual environment:
```bash
source myenv/bin/activate # Linux/macOS
.\myenv\Scripts\activate # Windows
```
Install the requirements:
```bash
pip install -r requirements.txt
```
Start the Django server:
```bash
python manage.py runserver
```
The server will start at http://127.0.0.1:8000/ .


---

> **Akshay R**,
>**Deepu Joseph**,
>*Semester 3, Masters in Computer Applications*,
>*College of Enginnering, Trivandrum*
>(*2023-2025*)