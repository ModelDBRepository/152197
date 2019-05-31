This simulation was used in the following article:

  Eguchi A, Neymotin SA, Stringer SM. (2014)
  Color opponent receptive fields self-organize in a biophysical model
  of visual cortex via spike-timing dependent plasticity.
  Front. Neural Circuits 8:16. doi: 10.3389/fncir.2014.00016

For questions email: akihiro dot eguchi at psy dot ox dot ac dot uk

This simulation was tested/developed on LINUX systems, but may run on
Microsoft Windows or Mac OS.

To run, you will need the NEURON simulator (available at
http://www.neuron.yale.edu) compiled with python enabled. To draw the
output you will need to have Matplotlib installed (
http://matplotlib.org/ ).

Instructions:
 Unzip the contents of the zip file to a new directory.

 compile the mod files from the command line with:
  nrnivmodl *.mod

The nrnivmodl command will produce an architecture-dependent folder
with a script called special.  On 64 bit systems the folder is
x86_64. To run the simulation from the command line use:
 python runMe.py

Various parameters used in the simulation are set in the python codes.
State of the networks are exported and saved every n iterations as
"Network_"+str(itr)+".obj" format so that various analysis can be
applied to the network with specific point during the training using
runMe2.py script.
