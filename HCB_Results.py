from InitiateLaunchSequence import *
import joblib

from sklearn.cluster import AgglomerativeClustering

ScriptBasename = os.path.basename(inspect.getfile(inspect.currentframe()))[:-3]
figlist = []

coords = numpy.array(joblib.load('/home/jandepue/fileserver/home/Documents/git/hydro-climatic-biomes/coords.pkl'))
theta = joblib.load('/mnt/HDS_ECOPROPHET/ECOPROPHET/SATEX/out/h_11_l_10/theta_h_11_l_10_iter3.pkl')
u = joblib.load('/mnt/HDS_ECOPROPHET/ECOPROPHET/SATEX/out/h_11_l_10/u_h_11_l_10_iter3.pkl')
v = joblib.load('/mnt/HDS_ECOPROPHET/ECOPROPHET/SATEX/out/h_11_l_10/v_h_11_l_10_iter3.pkl')
w = joblib.load('/mnt/HDS_ECOPROPHET/ECOPROPHET/SATEX/out/h_11_l_10/w_h_11_l_10_iter3.pkl')

Clust = AgglomerativeClustering(n_clusters=11, affinity='euclidean',)
Clust.fit(v)

Class = Clust.labels_

Class_names = {
	3: ['tropical','g'],
	1: ['transitional energy-driven',pyplot.get_cmap('Greens')(0.5)],
	8: ['transitional water-driven',pyplot.get_cmap('Greens')(0.9)],
	10:['subtropical energy-driven',pyplot.get_cmap('Oranges')(0.4)],
	6: ['subtropical water-driven',pyplot.get_cmap('Oranges')(0.7)],
	5: ['mid-latitude temperature-driven',pyplot.get_cmap('Purples')(0.4)],
	0: ['mid-latitude water-driven',pyplot.get_cmap('Purples')(0.7)],
	2: ['boreal energy-driven',pyplot.get_cmap('Blues')(0.2)],
	4: ['boreal temperature-driven',pyplot.get_cmap('Blues')(0.5)],
	7: ['boreal water/temperature-driven',pyplot.get_cmap('Blues')(0.7)],
	9: ['boreal water-driven',pyplot.get_cmap('Blues')(0.9)],
}


TowerRoot = '/mnt/HDS_SAFLAND/SAFLAND/ecoprophet/validation/towers'
SitesFilename = os.path.join(TowerRoot,'fluxmeta','thesites.csv')
Sites_DF = pandas.read_csv(SitesFilename,index_col = 1)

KoppenFilename = os.path.join(TowerRoot,'%s.csv'%'KoppenClassification')
Koppen_DF = pandas.read_csv(KoppenFilename,index_col = 0)
nF = Koppen_DF.shape[0]

Clust_DF = pandas.DataFrame()
for iF in range(nF):
	TestSite=Koppen_DF.index[iF]
	iMin = numpy.argmin((coords[:,1] - Sites_DF.loc[TestSite].towerlon)**2 \
						+ (coords[:,0] - Sites_DF.loc[TestSite].towerlat)**2)
	Clust_DF.loc[TestSite,'HCB_Code'] = Class[iMin]
	Clust_DF.loc[TestSite,'HCB_Class'] = Class_names[Class[iMin]][0]


fig = pyplot.figure(figsize=[20,10])
ax = fig.add_subplot(111)
# ax.scatter(coords[:,1],coords[:,0],4,Class,cmap='tab20')
for key in Class_names:
	filt = Class==key
	color = Class_names[key][1]
	ax.plot(coords[filt,1],coords[filt,0],'.',color=color,ms=4)
	ax.plot([],[],'o',color=color,ms=8,label=Class_names[key][0])
for iF in range(nF):
	TestSite=Koppen_DF.index[iF]
	ax.plot(Sites_DF.loc[TestSite].towerlon,Sites_DF.loc[TestSite].towerlat,'ro',mec='k')
ax.set_aspect('equal')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc=3)
figlist.append(fig)


Clust_DF.to_csv('%s.csv'%ScriptBasename)

#==============================================================================
# Write Plots
#==============================================================================

print('Writing plots'.center(50,'=')+'\n')

WriteRoot = '/home/jandepue/fileserver/home/Documents/PythonFigures'
basename = ScriptBasename
postfix = ''

# save plots to pdf
pdfname = os.path.join(WriteRoot, '%s%s.pdf'%(basename,postfix))
pp = PdfPages(pdfname)
for fig in figlist:
	pp.savefig(fig)

pp.close()

# # save plots to png
# extension='.png'
# for iF in range(len(figlist)):
#	 fig=figlist[iF]
#	 figname = os.path.join(Root, basename+'_%s'%iF + extension)
#	 fig.savefig(figname, bbox_inches='tight')

# pyplot.show()