# Template for all python projects 
This is a template for Python Projects in Goldenberg Lab. There are three folders: 

- **data** All of the data stored in this repo should be located in this folder.
- **processing** All of the code designed for processing should be saved in this folder. 
- **analysis_modeling** should be used for analysis of processed data. 

## Requirements before running python code
This is a template for Python Projects in Goldenberg Lab. To use this template, please:

- **1.** Download Anaconda and Python. We recommend starting with the latest Python version and Anaconda version. The tutorial [here](https://docs.jupyter.org/en/latest/install/notebook-classic.html) is helpful.
- **2.** Anaconda will automatically install Jupyter notebook. We recommend using Jupyter notebook for the analysis because it allows users to easily run code with instant feedback. We have also created an `environment_set_up_activation.ipynb` Jupyter notebook that will help you create your working environment.

### Use this template for the first time (if you are not replicating/ adding on to an existing analysis)

- Choose the repository's username/organization name.
    - Please set the owner of your analysis template to our lab (`GoldenbergLab`).
    - All analysis repositories should start as public, unless indicated otherwise.
- Name the repository following the lab naming convention. Full guide on repository naming conventions can be found [here](https://github.com/GoldenbergLab/naming-conventions#repository-names). In short, github repositories are following this naming convention: `project-name-analysis`. So if your project is about counting kittens, your repository name is `counting-kittens-analysis`.
- Add a description of your project. Please include:
    1. Project Name
    2. Date of repository creation
    3. Your name, and the names of other who worked on it
    4. The purpose of the project and the main question you asked
    5. The source of the data for the analysis (Prolific, MTURK, Qualtrics, etc.)
- Once the repository is generated, you will be redirected to it.

## Setting up the enviroment 

- **1.** To set up a NEW enviroment to run the code, open the `Creating a new environment.ipynb` notebook file. Once you finish working on your analysis, make sure to run `Exporting and saving your environment.ipynb` to update your environment with all of the packages.
   
- **2.** To REPRODUCE an EXISTING ENVIRONMENT (if you are replicating code or expanding upon an already-existing analysis) navigate to the `Reproduce an existing environment.ipynb` file

Once you have completed the steps designated in the file, you will have the environment set up and activated, ready to use for your project.

### Clone the repository to your local machine

To connect (or clone) the analysis repository from GitHub (aka the internet) to your own computer, please go to the green `Code` button with a dropdown menu and do one of two things:

1. **Open with [GitHub Desktop](https://desktop.github.com/)** (for those with less coding experience)

- Please download GitHub desktop using the link above
- Click `Open with GitHub Desktop`, and pick a location in your personal file system to store it using the GitHub Desktop cloning interface. The default is `Documents\GitHub` (this is fine to use).
- The folder should appear now in `Documents\GitHub\your-repository-name`.
- If using Linux, see option 2.

2. **Command line using HTTPS method** (for those with more coding experience)

- If using Linux or prefer the command line, you will use the HTTPS-based URL from the Code dropdown menu. The URL will look like https://github.com/startyourlab/r-project-template.git.
- Click the clipboard icon to copy it. From within your projects directory in the terminal, run git clone ..., replacing the "..." with the URL you just copied.


## Using python enviroment

In Python, an environment refers to a self-contained and isolated workspace that contains all the necessary dependencies, libraries, and configurations to run Python code and execute projects.

Python environments are used to manage different versions of Python, as well as the packages and dependencies associated with a specific project. They provide a way to keep project dependencies separate and prevent conflicts between different projects or versions.

Commonly used tools for managing Python environments include Anaconda/Miniconda and virtualenv. These tools allow you to create, manage, and switch between different environments easily.

Python environments offer the following benefits:

- **Dependency Isolation**: Each environment can have its own set of packages and dependencies, allowing you to control and manage the specific versions required for your project.

- **Reproducibility**: Environments enable you to reproduce your project's setup on different machines, ensuring consistent behavior and results.

- **Project Organization**: By using separate environments for different projects, you can keep project-specific dependencies separate and avoid conflicts between different projects.

- **Collaboration**: Environments make it easier to collaborate with others by providing a standardized setup that can be shared and replicated.

To create a Python environment, you typically define its specifications in a configuration file, such as `environment.yml` or `requirements.txt`, which lists the required packages and their versions. You can then use a package manager like conda or pip to create and manage the environment based on the configuration file.

Managing Python environments is especially useful when working on complex projects, collaborating with others, or deploying your code in different environments.

---
For more information on managing Python environments in anaconda refer to [this](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

