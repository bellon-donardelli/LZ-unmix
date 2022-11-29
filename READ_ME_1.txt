This is a quick tutorial to install the required libraries to run LZ-unmix.

	1. Firstly, download the zip file containing the LZ-unmix.ipyfile and the LZ_unmix_functions.py file.

	2. Put these files together in a folder of your preference.

	3. LZ-unmix is, essentially, an ipynb-file. So Jupyter Notebook is mandatory! Go ahead to https://www.anaconda.com/ and download and execute
	the ANACONDA graphical installer (depending on your operational system).

	4. Open the Anaconda prompt and type (sequentially, press ENTER):
		pip install numpy matplotlib scipy pandas ipywidgets seaborn plotly

	5. Close the Anaconda prompt, reopen it and type:
		jupyter notebook
	
	6. After launching Jupyter Notebook, search for your folder and double click on the LZ-unmix.ipyfile to open it. 

	7. The environment is set to run your models!

PS: Always make sure to have your csv-data file within the same folder where LZ-unmix is.