import os
import pprint
import random
import sys
import wx
import csv
import io
# The recommended way to use wx with mpl is with the WXAgg
# backend. 
#
import matplotlib
matplotlib.use('WXAgg')
import numpy as np
import pylab

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from sklearn.svm import SVC,LinearSVC

from matplotlib.figure import Figure
import wx.lib.scrolledpanel
class ImportDatafile:
	""" Getting columns of data from csv file and build a dictionary
	"""
	def __init__(self,infile):
		self.infile = infile

	def FromVerticalCSV(self):
		DataSet = {}
		titleset = []
		ftest = open(self.infile,'r')
		_n = 0
		strftest = ''
		for line in ftest:
			strftest += line
			_n += 1
			if _n>1:
				break
		strftest = strftest.split('\n')
		strftest = strftest[:-1]
		if  len(strftest)<2:
			raise Exception("Data file "+self.infile+" is too small for analysis!")

		_posdelimiters = ',;:|\t '
		gooddelimiter=';'
		for p in _posdelimiters:
			if len(strftest[0].split(p))==len(strftest[1].split(p)):
				if len(strftest[0].split(p))!=1:
					gooddelimiter= p
		DataSet = np.loadtxt(open(self.infile,"rb"),delimiter=gooddelimiter,skiprows=1)
		DataSet = DataSet.transpose()
		titleset=strftest[0].split(gooddelimiter)
		return [DataSet,titleset]

########################################################################
class RandomPanel(wx.Panel):
	""""""

	#----------------------------------------------------------------------
	def __init__(self, parent, color):
		"""Constructor"""
		wx.Panel.__init__(self, parent)
		self.SetBackgroundColour(color)



########################################################################
class MainPanel(wx.Panel):
	""""""

	#----------------------------------------------------------------------
	def __init__(self, parent):
		"""Constructor"""
		wx.Panel.__init__(self, parent)

		topSplitter = wx.SplitterWindow(self)
		hSplitter = wx.SplitterWindow(topSplitter)
		bSplitter = wx.SplitterWindow(topSplitter)

		self.panelOne = RandomPanel(hSplitter, "gray")
		self.panelTwo = RandomPanel(hSplitter, "gray")
		hSplitter.SplitVertically(self.panelOne, self.panelTwo,-650)
		# hSplitter.SetSashGravity(0.5)

		self.panelThree = RandomPanel(bSplitter, "gray")
		self.panelFour = RandomPanel(bSplitter, "blue")
		bSplitter.SplitVertically(self.panelThree,self.panelFour,-650)
		topSplitter.SplitHorizontally(hSplitter, bSplitter,500)
		# topSplitter.SetSashGravity(0.5)

		self.fbutton = wx.Button(self.panelOne, -1, "Import 1st .csv",pos=(15,30),size=(120,30))
		self.fbutton.Bind(wx.EVT_BUTTON, self.OnButton)



		self.varlistbox = wx.ListBox(choices=[], id=wx.NewId(), name='varlistbox', parent=self.panelOne, pos=(10, 110), size=wx.Size(140, 260), style=0)
		self.varlistbox.Bind(wx.EVT_LISTBOX, self.OnSelect)


		self.fbutton2 = wx.Button(self.panelOne, -1, "Import 2nd .csv",pos=(15,64),size=(120,30))
		self.fbutton2.Bind(wx.EVT_BUTTON, self.OnButton2)


		self.varlistbox2 = wx.ListBox(choices=[], id=wx.NewId(), name='varlistbox2', parent=self.panelOne, pos=(160, 110), size=wx.Size(184, 260), style=0)
		self.varlistbox2.Bind(wx.EVT_LISTBOX, self.OnSelect2)

		self.possible_kernels = ['Kernel = '+k for k in ['rbf','linear']]
		self.kernelchoicebox = wx.ComboBox(value=self.possible_kernels[0],choices=self.possible_kernels, id=wx.NewId(), name='kernelchoicebox', parent=self.panelThree, pos=(160, 110), size=wx.DefaultSize)
		self.kernelchoicebox.Bind(wx.EVT_COMBOBOX, self.OnSelectKernel)
		self.chosenkernel = self.possible_kernels[0].split('=')[-1].replace(' ','')

		self.possible_cvalues = ['C = '+str((.00001*(10**_c))) for _c in range(11)]
		self.cvaluechoicebox = wx.ComboBox(value=self.possible_cvalues[0],choices=self.possible_cvalues, id=wx.NewId(), name='cvaluechoicebox', parent=self.panelThree, pos=(160, 140), size=wx.DefaultSize)
		self.cvaluechoicebox.Bind(wx.EVT_COMBOBOX, self.OnSelectCvalue)
		self.chosencvalue = self.possible_cvalues[0].split('=')[-1].replace(' ','')



		self.varlistbox3 = wx.ListBox(choices=[], id=wx.NewId(), name='varlistbox3', parent=self.panelThree, pos=(10, 110), size=wx.Size(140, 260), style=wx.LB_MULTIPLE)
		self.varlistbox3.Bind(wx.EVT_LISTBOX, self.OnSelect3)



		self.selindex = 0
		self.compindex = 0
		self.selected_variables = []
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(topSplitter, 1, wx.EXPAND)
		self.SetSizer(sizer)


		self.dataset2 = None
		self.infile=None
		self.infile2=None

	def OnButton(self, evt):
		# print ' OnButton selection '
		dlg = wx.FileDialog(
			self, message="Choose a file",
			defaultDir=os.getcwd(),
			defaultFile="",
			style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
			)

		self.infile = None
		if dlg.ShowModal() == wx.ID_OK:
			paths = dlg.GetPaths()
			infile =  str(paths[0])
			self.infile=infile
		self.fbutton.SetBackgroundColour('lightblue')	
		[self.dataset,self.titleset] = ImportDatafile(infile).FromVerticalCSV()
		self.varlistbox.Clear()
		for vartitle in self.titleset:
			self.varlistbox.Append(vartitle)
			self.varlistbox3.Append(vartitle)


	def OnButton2(self, evt):
		print ' OnButton selection '
		dlg = wx.FileDialog(
			self, message="Choose a file",
			defaultDir=os.getcwd(),
			defaultFile="",
			style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
			)

		self.infile2 = None

		if dlg.ShowModal() == wx.ID_OK:
			paths = dlg.GetPaths()
			self.infile2 =  str(paths[0])
		self.fbutton2.SetBackgroundColour('pink')	

		[self.dataset2,self.titleset2] = ImportDatafile(self.infile2).FromVerticalCSV()
	   


	def OnSelect(self, evt):
		self.selindex = evt.GetSelection()
		self.varlistbox2.Clear()
		for vartitle in self.titleset:
			if vartitle != self.titleset[self.selindex]:
				self.varlistbox2.Append('Plot vs: '+vartitle)
			else:
				self.varlistbox2.Append('Histogram '+self.titleset[self.selindex])



	def OnSelect2(self, evt):

		self.figure = Figure(figsize=(6, 4.5), dpi=100, facecolor='w', edgecolor='k')
		self.mainaxis = self.figure.add_subplot(1,1,1)

		self.canvas = FigureCanvas(self.panelTwo, -1, self.figure)
		self.compindex = evt.GetSelection()
		if self.selindex==self.compindex:
			self.mainaxis.hist(self.dataset[self.selindex], 50, edgecolor='blue',alpha=0.75,label=(self.infile).split('.')[0].split('/')[-1],histtype='step')
			self.mainaxis.set_xlabel(self.titleset[self.selindex])
			self.mainaxis.set_ylabel('Probability')
			if self.infile2!=None:
				self.mainaxis.hist(self.dataset2[self.selindex], 50,edgecolor='red',alpha=0.75,label=(self.infile2).split('.')[0].split('/')[-1],histtype='step')


		else:
			self.mainaxis.scatter(self.dataset[self.selindex],self.dataset[self.compindex],marker='.',s=1,facecolor='0.5', edgecolor='0.5',label=(self.infile).split('.')[0].split('/')[-1])
			self.mainaxis.set_xlabel(self.titleset[self.selindex])
			self.mainaxis.set_ylabel(self.titleset[self.compindex])
			fit_1 = np.polyfit(self.dataset[self.selindex],self.dataset[self.compindex],1)
			coef_1 = np.poly1d(fit_1)
			fit_2 = np.polyfit(self.dataset[self.selindex],self.dataset[self.compindex],2)
			coef_2 = np.poly1d(fit_2)

			x_min = min(self.dataset[self.selindex])
			x_max = max(self.dataset[self.selindex])
			xspace = [x_min + 0.01*(x_max-x_min)*interval for interval in range(100)]
			y_1 = coef_1(xspace)
			y_2 = coef_2(xspace)

			# self.mainaxis.set_grid(True)
			self.mainaxis.plot(xspace,y_1,'red', label='Linear Fit')			
			self.mainaxis.plot(xspace,y_2,'blue', label='Quadratic Fit')			
		legend = self.mainaxis.legend(loc='upper left', shadow=True)
		for alabel in legend.get_texts():
			alabel.set_fontsize('small')		


		self.canvas.draw()		
		self.canvas.resize(10,50)



	def DoMVA(self):

		self.figure2 = Figure(figsize=(6, 4.5), dpi=100, facecolor='w', edgecolor='k')
		self.mainaxis2 = self.figure2.add_subplot(1,1,1)

		self.canvas2 = FigureCanvas(self.panelFour, -1, self.figure2)

		selected_variables = self.selected_variables
		_S= self.dataset[selected_variables,:1000]
		_B= self.dataset2[selected_variables,:1000]

		v = selected_variables[0]

		if self.dataset2 != None or _S.shape[0]<=1:
			if len(selected_variables)==1:
				v = selected_variables[0]
				self.mainaxis2.hist(_S[0,:], 50,alpha=0.75,label=('Training Subset: '+self.infile.split('/')[-1].split('.')[0]),histtype='step')
				self.mainaxis2.hist(_B[0,:], 50,edgecolor='red',alpha=0.75,label=('Training Subset: '+self.infile2.split('/')[-1].split('.')[0]),histtype='step')

				self.mainaxis2.set_xlabel(self.titleset[self.selindex])
				self.mainaxis2.set_ylabel('Probability')
				legend2 = self.mainaxis2.legend(loc='upper left', shadow=True)


				for alabel in legend2.get_texts():
					alabel.set_fontsize('small')


		if _S.shape[0]>1:

			_ST = np.ones((_S.shape[-1]))
			_BT = -1*np.ones((_B.shape[-1]))
			_X = np.concatenate((_S,_B),axis=1)
			_Y = np.concatenate((_ST,_BT),axis=0)
			_X=_X.transpose()

			print 'USING:',self.chosenkernel
			svm = SVC(C = float(self.chosencvalue), kernel = self.chosenkernel)
			svm.fit(_X,_Y)

			_S_TrainHist = svm.decision_function(_S.transpose()).transpose()[0]
			_B_TrainHist = svm.decision_function(_B.transpose()).transpose()[0]
			self.mainaxis2.hist(_S_TrainHist, 50,alpha=0.75,label=(self.infile.split('/')[-1].split('.')[0]),histtype='step')
			self.mainaxis2.hist(_B_TrainHist, 50,edgecolor='red',alpha=0.75,label=(self.infile2.split('/')[-1].split('.')[0]),histtype='step')

			self.mainaxis2.axvline(x=0,color='black',linestyle='--',alpha=0.2)

			_xmax = 1.2*max([_S_TrainHist.max(),_B_TrainHist.max()])
			_xmin = -1.2*(abs(1.0*min([_S_TrainHist.min(),_B_TrainHist.min()])))		
			self.mainaxis2.axvspan(0.0, _xmax, color='blue',alpha=0.08)
			self.mainaxis2.axvspan(_xmin,0.0, color='red',alpha=0.08)

			self.mainaxis2.set_xlim(_xmin,_xmax  )

			self.mainaxis2.set_xlabel('Machine Learning Classifier')
			self.mainaxis2.set_ylabel('Probability')
			legend2 = self.mainaxis2.legend(loc='upper left', shadow=True)

			

			for alabel in legend2.get_texts():
				alabel.set_fontsize('small')


		self.canvas2.draw()		
		self.canvas2.resize(10,50)




	def OnSelectKernel(self,evt):
		# print "Rechoosing Kernel"
		self.chosenkernel = self.possible_kernels[evt.GetSelection()].split('=')[-1].replace(' ','')
		# print self.chosenkernel
		if len(self.selected_variables)>0:
			self.DoMVA()

	def OnSelectCvalue(self,evt):
		self.chosencvalue = self.possible_cvalues[evt.GetSelection()].split('=')[-1].replace(' ','')
		# print self.chosenkernel
		if len(self.selected_variables)>0:
			self.DoMVA()


	def OnSelect3(self, evt):



		self.selected_variables = []
		for varindex in range(len(self.titleset)):
			if self.varlistbox3.IsSelected(varindex):
				self.selected_variables.append(varindex)

		if len(self.selected_variables)>0:
			self.DoMVA()



########################################################################
class MainFrame(wx.Frame):
	""""""

	#----------------------------------------------------------------------
	def __init__(self):
		"""Constructor"""
		wx.Frame.__init__(self, None, title="Nested Splitters",
						  size=(1000,1000))
		panel = MainPanel(self)
		self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
	app = wx.App(False)
	frame = MainFrame()
	app.MainLoop()