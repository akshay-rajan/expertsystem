![alt text](./others/screenshot.png)

Machine Learning is a vast and complex field that requires a lot of dedication and time to master. 
However, the complexity of ML algorithms and the necessity for programming skills create
significant barriers for individuals who wish to leverage ML for their projects or research. 
But what if we could remove this constraint and allow the users to build models without any coding knowledge?

**Expert System** is a platform that allows users to build, train, and test Machine Learning Models without the need for programming languages. 

It provides a user-friendly interface where users can:

![Select Category](./others/screenshot2.png)
![Select Algorithm](./others/classification.png)
![Process Data](./others/preprocessing.png)
![Upload Data](./others/screenshot3.png)

And finally, see the 

![Process Data](./others/screenshot4.png)

in just a few clicks.

Expert System is designed to make Machine Learning accessible to everyone, regardless of their coding knowledge. 

### Flow

```mermaid
flowchart LR
    A(Start) --> B[Preprocessing]

    A --> D(Modelling)
    B --> D
    D --> E[Select Algorithm]
    E --> F[Pick Features, Target etc.]
    F --> G{Training}
    G --> H[Evaluation]
    G --> I[Prediction]
    G --> J[View Code]
    G --> K[Download Model]

    H --> Z(End)
    I --> Z
    J --> Z
    K --> Z    
```


### *Technologies Used*

![Static Badge](https://img.shields.io/badge/-Django-darkgreen?style=for-the-badge&logo=django)
![Static Badge](https://img.shields.io/badge/-JavaScript-white?style=for-the-badge&logo=javascript)
![Static Badge](https://img.shields.io/badge/-d3.js-orange?style=for-the-badge&logo=javascript)
![Static Badge](https://img.shields.io/badge/-plotly.js-navy?style=for-the-badge&logo=javascript)
![Static Badge](https://img.shields.io/badge/-Scikit_Learn-blue?style=for-the-badge&logo=scikit-learn)
![Static Badge](https://img.shields.io/badge/-pandas-purple?style=for-the-badge&logo=pandas)
![Static Badge](https://img.shields.io/badge/-numpy-cyan?style=for-the-badge&logo=numpy)
![Static Badge](https://img.shields.io/badge/-prism.js-black?style=for-the-badge&logo=javascript)
![Static Badge](https://img.shields.io/badge/-bootstrap-white?style=for-the-badge&logo=bootstrap)

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
Run the database migrations:
```bash
python manage.py migrate
```
Start the Django server:
```bash
python manage.py runserver
```
The server will start at http://127.0.0.1:8000/ .


---

<a href="https://github.com/akshay-rajan/expertsystem/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=akshay-rajan/expertsystem" />
</a>

> **Akshay R**,
>**Deepu Joseph**,
>*Semester 3, Masters in Computer Applications*,
>*College of Enginnering, Trivandrum*
>(*2023-2025*)

