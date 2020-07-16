import urllib2
import numpy as np
import matplotlib.pyplot as plt
from ftplib import FTP
from zipfile import ZipFile
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


#script to import data from the Metaboligths repository (http://www.ebi.ac.uk/metabolights/)
#the datasets to be analysed comprise a 1H-NMR metabolic profiling study of human urine from type-2 diabetic patients and healthy controls
#the study identifier is 'MTBLS1'; the code should be applicable to any Metaboligths NMR study with a similar data structure 
#the main function, metNMR('study_identifier') perfoms the full analysis and ploting


def read_file_list(study_identifier):
	"""this function reads the Metaboligths project file containing the description of NMR experiments and analysis 
	and returns the filenames of the zipped spectral folders"""
	BASE_URL = 'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/'
	identifier = study_identifier.lower()
	response = urllib2.urlopen(BASE_URL+study_identifier+'/'+'a_'+identifier+'_metabolite_profiling_NMR_spectroscopy.txt')
	file_list = np.genfromtxt(response, delimiter ='"\t"', usecols = (35,36), skip_header = 1, dtype=('string'))
	return file_list

def fetch_files(study_identifier):
	"""This function performs the download of the datasets according to the study identifier provided"""
	file_list = read_file_list(study_identifier)
	ftp = FTP('ftp.ebi.ac.uk')   # connect to host, default port
	ftp.login()            # user anonymous, passwd anonymous@
	ftp.cwd('pub/databases/metabolights/studies/public/'+study_identifier+'/')
	for file in file_list[:,1]:
		ftp.retrbinary('RETR '+file, open(file, 'wb').write)
		print 'Downloading '+file
	ftp.quit()
	
def read_intensities(file_path):	
	"""this function just read the intensities of one processed spectrum which is a binary file containing only real numbers (1r)"""
	arr = np.fromfile(file_path,dtype=np.dtype('<i4'))
	return arr

def read_spectra(study_identifier):
	"""extracts .zip and reads Bruker 1d spectra folder and creates an array of intensities, one line per spectrum"""
	file_list = read_file_list(study_identifier)
	zip_file = ZipFile(file_list[0,1], 'r')
	local = file_list[0,0]+'/10/pdata/1/1r'
	file_path = zip_file.extract(local)
	arr = read_intensities(file_path)
	arr_c = np.zeros((len(file_list),len(arr)))
	arr_c[0,:] = arr
	for ix in range(1,len(file_list)):
		zip_file = ZipFile(file_list[ix,1], 'r')
		local = file_list[ix,0]+'/10/pdata/1/1r'
		file_path = zip_file.extract(local)
		arr_c[ix,:] = read_intensities(file_path)
	return arr_c

def read_ppm(study_identifier):
	"""this function reads the necessary spectral properties to calculate the chemical shift scale in ppm"""
	file_list = read_file_list(study_identifier) # unzip the procs file for the ppm scale calculation
	zip_file = ZipFile(file_list[0,1], 'r') #only only ppm scale needs to be read, only the first spectrum in the list is used
	procs = file_list[0,0]+'/10/pdata/1/procs'
	file_path = zip_file.extract(procs)
	procpars = open(file_path).readlines() #the parameters needed for ppm calculation are discribed bellow
	for line in procpars:
		if line.startswith('##$OFFSET='):
			offset = float(line[10:19]) #offset: the center of the spectrum in ppm
		if line.startswith('##$SF='):
			sf = float(line[6:23]) #sf: spectrometer frequency in Hertz
		if line.startswith('##$SW_p='):
			sw = float(line[8:25]) #sw: sweep-width in Hertz
		if line.startswith('##$SI='): 
			si = float(line[6:12]) #si: total number of spectral points
	swp = sw/sf
	dppm = swp/(si)
	ppm = np.arange(offset,(offset-swp),-dppm)
	return ppm

def read_cal_spectra(study_identifier):
	"""calibration of the chemical shift scale since the scale of the first spectrum is considered, so this function automatically 
	reads and calibrates the chemical shift of the reference resonance (TSP) to 0 ppm""" 
	spec = read_spectra(study_identifier) # creates the spectra array
	ppm = read_ppm(study_identifier) #creates the ppm scale
	lim1 = ppm > -0.02 
	lim2 = ppm <= 0.02
	lims = np.logical_and(lim1,lim2) #limits where the TSP resonance should be contained
	target = np.where(ppm==np.median(ppm[lims])) #the index of the ppm corresponding approximately to 0 ppm
	new_spec = spec
	last = len(ppm)
	for ix in range(0,np.shape(spec)[0]):
		tsp_max = np.where(spec[ix,:]==max(spec[ix,lims]))
		diff = abs(int(tsp_max[0]-target[0]))
		if tsp_max > target:
			new_spec[ix,:] = np.concatenate((spec[ix,diff:last],spec[ix,0:diff]))
		if tsp_max < target:
			new = last-diff
			new_spec[ix,:] = np.concatenate((spec[ix,new:last],spec[ix,0:new]))
		if tsp_max == target: 
			new_spec[ix,:] = spec[ix,:]
	return (new_spec,ppm)

	
def trim(spec,ppm):
	"""creates a spectral array without redundant spectral regions: residual water and urea some baseline regions are removed"""
	ind1 = np.median(np.where(np.around(ppm,2)==10.02))
	ind2 = np.median(np.where(np.around(ppm,2)==6.27))
	ind3 = np.median(np.where(np.around(ppm,2)==5.53))
	ind4 = np.median(np.where(np.around(ppm,2)==4.90))
	ind5 = np.median(np.where(np.around(ppm,2)==4.67))
	ind6 = np.median(np.where(np.around(ppm,2)==0.21))
	t_spec = np.hstack((spec[:,ind1:ind2],spec[:,ind3:ind4],spec[:,ind5:ind6]))
	t_ppm = np.concatenate((ppm[ind1:ind2],ppm[ind3:ind4],ppm[ind5:ind6]))
	return (t_spec, t_ppm)

def pqn(spec):
	"""normalizes the spectral intensities according to the method of Probability Quotient Normalization from 
	Dieterle et al Anal Chem 2006,78(13):4281-90""" 
	med = np.median(spec,axis=0)
	norm_spec = spec
	for ix in range(0,len(med)):
		if med[ix]==0:
			med[ix] = 0.01
	for ix in range(0,spec.shape[0]):
		meds = spec[ix,:]/med
		quotient = np.median(meds)
		norm_spec[ix,:] = spec[ix,:]/quotient
	return norm_spec

def total_area(spec):
	"""normalizes the spectral intensities to the sum of all the spectral intensities (total area)""" 
	norm_spec = spec
	for ix in range(0,spec.shape[0]):
		total = np.sum(spec[ix,:])
		norm_spec[ix,:] = spec[ix,:]/total
	return norm_spec
	
def pareto(spec):
	"""performs pareto scaling on the NMR normalized data"""
	sd = np.std(spec,axis=0)
	root_sd = np.sqrt(sd)
	scaled_spec = spec
	for ix in range(0,spec.shape[0]):
		scaled_spec[ix,:] = spec[ix,:]/root_sd
	return scaled_spec

	
def read_meta(study_identifier):
	"""function to read the Metaboligths project info file and take sample names from the 8th column, 
	gender from the 9th and sample group from the 12th"""
	BASE_URL = 'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/'
	response = urllib2.urlopen(BASE_URL+study_identifier+'/'+'s_'+study_identifier+'.txt')
	meta_data = np.genfromtxt(response, delimiter = '"	"', skip_header = 1 , usecols = (8,9,12), dtype=None) 
	return meta_data
	
def plot(spec,ppm,filename):	
	"""this small function generates the stacked plot of all spectra"""
	for ix in range(0,np.shape(spec)[0]):
		plt.plot(ppm,spec[ix,:])
	plt.xlim(max(ppm),min(ppm))
	plt.xlabel('Chemical Shift (ppm)')
	plt.ylabel('Intensity (AU)')
	plt.savefig(filename+'.png')
	plt.clf()	
	
#this function plots all the spectra that were loaded into the an array;
#notice that the calibration of the chemical shift scale is lost; it needs to be re-calibrated using an external function 
def plot_spectra(study_identifier):	
	spec = read_spectra(study_identifier) # creates the spectra array
	ppm = read_ppm(study_identifier)
	for ix in range(0,np.shape(spec)[0]):
		plt.plot(ppm,spec[ix,:])
	plt.xlim(14,-5)
	plt.xlabel('Chemical Shift (ppm)')
	plt.ylabel('Intensity (AU)')
	plt.show()
	plt.close()

def pca(spec,meta):
	"""perform pca on the object spectra and return the corresponding scores and loadings, assuming 2 components"""
	pca = PCA(n_components=2)
	scores = pca.fit_transform(spec)
	loadings = pca.components_
	exp_var = np.around(pca.explained_variance_ratio_*100,1)
	var = np.array_str(exp_var) #percentages of variance explained by each component are converted to stings to use in the plot labels
	m_mask = np.where(meta[:,1]=='Male') #masks for gender selection
	f_mask = np.where(meta[:,1]=='Female')
	dm_mask = np.where(meta[:,2]=='diabetes mellitus') #masks for group selection
	c_mask = np.where(meta[:,2]=='Control Group')
	plt.figure(1)
	plt.subplot(121)
	plt.scatter(scores[m_mask,0],scores[m_mask,1],color='blue')
	plt.scatter(scores[f_mask,0],scores[f_mask,1],color='red')
	plt.xlabel(('PC1'+' '+'('+var[2:6]+'%'+')'))
	plt.ylabel(('PC2'+' '+'('+var[8:12]+'%'+')'))
	plt.title('Male '+'${vs}$'+' Female subjects',fontsize=12)
	plt.legend(('Male','Female'),scatterpoints=1,fontsize=12)
	plt.subplot(122)	
	plt.scatter(scores[c_mask,0],scores[c_mask,1],color='black')
	plt.scatter(scores[dm_mask,0],scores[dm_mask,1],color='magenta')
	plt.xlabel(('PC1'+' '+'('+var[2:6]+'%'+')'))
	plt.title('Healthy '+'${vs}$'+' Diabetic subjects',fontsize=12)
	plt.legend(('Healthy','Diabetic'),scatterpoints=1,fontsize=12)
	plt.savefig('pca.png')
	plt.clf()

def plsda(spec,ppm,meta):	
	"""performs pls-da regression on the normalized and scaled spectra against the know class labels;two components are computed"""
	X = spec
	Y = spec[:,0]
	for ix in range(0,len(Y)):
		if meta[ix,2] == 'diabetes mellitus':
			Y[ix] = 1
		if meta[ix,2] == 'Control Group':
			Y[ix] = 0
	pls = PLSRegression(n_components=2,scale=False)
	pls.fit(X, Y)
	scores = pls.x_scores_ 
	loadings = pls.x_weights_ 
	dm_mask = np.where(meta[:,2]=='diabetes mellitus') #masks for group selection
	c_mask = np.where(meta[:,2]=='Control Group')
	plt.scatter(scores[c_mask,0],scores[c_mask,1],color='black')
	plt.scatter(scores[dm_mask,0],scores[dm_mask,1],color='magenta')
	plt.xlabel('LV1')
	plt.ylabel('LV2')
	plt.title('Healthy '+'${vs}$'+' Diabetic subjects')
	plt.legend(('Healthy','Diabetic'),scatterpoints=1,fontsize=12)
	plt.savefig('pls_scores.png')
	plt.clf()
	plt.plot(ppm,loadings[:,0], color='black')
	plt.xlabel('Chemical Shift (ppm)')
	plt.ylabel('weights(LV1)')
	plt.title('Healthy '+'${vs}$'+' Diabetic subjects')
	plt.xlim(10,0.2)
	plt.ylim(-0.05,0.05)
	plt.savefig('pls_weights.png')
	plt.clf()
	
	
#the main function
def metNMR(study_identifier):
	"""this function automates the download, processing and analysis procedure"""
	fetch_files(study_identifier) #downloads the folders containing the spectra via FTP
	spec,ppm = read_cal_spectra(study_identifier) #reads and calibrates the spectra intensities and the ppm scale
	plot(spec,ppm,'full_spectra')
	t_spec,t_ppm = trim(spec,ppm) #removes the resonances of HOD, urea and
	plot(t_spec,t_ppm,'trimmed_spectra')
	norm_spec = total_area(t_spec) #normalizes the spectra according to the total area
	scaled_spec = pareto(norm_spec) #scales the spectra using pareto scaling
	meta = read_meta(study_identifier) #sample name, gender and group information are obtained
	pca(scaled_spec,meta) #performs the PCA and plots two scores plots according to gender and group 
	plsda(scaled_spec,t_ppm,meta) #performs PLS-DA on group information; plots scores and weights for the separating component
	print "Analysis complete!"
	
#additional functions usefull for future developments and not used here
	
#this is used for spectra alignment	using the icoshift algorithm by Savorani et al Journal of Magnetic Resonance 202 (2010) 190–202 and coded in Python by Martin Fitzpatrick
#future developments will make use of other algoritms such as CluPA(from R), RSPA (from Matlab) or PCANS (Python)
def align_spec(study_identifier):
	"""aligns spectra using icoshift algorithm"""
	new_spec, new_ppm = trim(study_identifier)
	xCS,ints,ind,target = ics.icoshift('average',new_spec)
	return (xCS,ints,ind,target,new_ppm)

#this function is not be used since the can lead to significant loss of spectral resolution which makes interpretation difficult
def binning(spec,ppm,k):	
	"""performs the average every k consecutive points in order to reduce shifts in ppm between samples"""
	b_ppm = ppm.reshape(-1,k).mean(axis=1)
	b_spec = np.zeros((np.shape(spec)[0],len(ppm)/k))
	for ix in range(0,np.shape(spec)[0]):
		b_spec[ix,:] = spec[ix,:].reshape(-1,k).mean(axis=1)
	return (b_spec,b_ppm)