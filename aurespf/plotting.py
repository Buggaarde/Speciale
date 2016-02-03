#! /usr/bin/env python
import numpy as np
from pylab import plt
import multiprocessing as mp
from regions.tools import *
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import matplotlib as mpl

#test comment
##################################################
### common Plotting definitions of size and colour 
##################################################
##################################################
##################################################
def ann(r):
    if r==0: return 30
    return (1-(1+(r/100.0))**-30)/(r/100.0)

def Energy(r=4.0):
    return 3027140662*ann(r)



EUR = unichr(0x20AC)   
colwidth = (3.425)
dcolwidth = (2*3.425+0.236) 

blue = '#134b7c'
yellow = '#f8ca00'
orange = '#e97f02'
brown = '#876310'
green = '#4a8e05'
lightgreen = '#b9f73e'#'#c9f76f'
red = '#ae1215'
purple = '#4f0a3d'
darkred= '#4f1215'
pink = '#bd157d'
lightpink = '#d89bc2'
aqua = '#47fff9'
darkblue = '#09233b'
lightblue = '#8dc1e0'
grayblue = '#4a7fa2'

blue_cycle = [darkblue, blue, grayblue, lightblue]

long_blue_cycle = []

color_cycle = [blue, red, orange, purple, green, pink, lightblue, darkred, yellow]

au_cdict = {'red': ((0.0,int(yellow[1:3],16)/255.0,int(yellow[1:3],16)/255.0),
(0.5,int(green[1:3],16)/255.0,int(green[1:3],16)/255.0),
(1.0,int(blue[1:3],16)/255.0,int(blue[1:3],16)/255.0)),
'green': ((0.0,int(yellow[3:5],16)/255.0,int(yellow[3:5],16)/255.0),
(0.5,int(green[3:5],16)/255.0,int(green[3:5],16)/255.0),
(1.0,int(blue[3:5],16)/255.0,int(blue[3:5],16)/255.0)),
'blue': ((0.0,int(yellow[5:7],16)/255.0,int(yellow[5:7],16)/255.0),
(0.5,int(green[5:7],16)/255.0,int(green[5:7],16)/255.0),
(1.0,int(blue[5:7],16)/255.0,int(blue[5:7],16)/255.0))}

au_cmap = LinearSegmentedColormap('au_cmap',au_cdict,256)




plt.rc('lines', lw=2)
plt.rc('lines', dash_capstyle = 'round')
plt.rc('axes', color_cycle = color_cycle)
plt.rc('legend', labelspacing = 0.25, borderpad = .2)


def histogramatron(L,S,title,path = '', noload=0): #M=today, N=intermediate, O=99P
    x0=-1.5#-2
    x1=2.001#3.001
    b=np.arange(x0,x1,(x1-x0)/250.)
    
    plt.close()
    ax=plt.subplot(1,1,1)
    long_len = len(S[0][0])
    if not noload:
        plt.plot(b[0:-1], plt.hist(non_zeros(-L),bins=b,normed=1,visible=0)[0], label = "Load", color = "k")
    for s in S:
        (serie, farve, navn) = s
        serie  = non_zeros(serie)
        x = 1.0
        if S[0][2] == 'Europe' and 'Germany' in navn: x=4.0
        ax.plot(b[0:-1], plt.hist(non_zeros(serie),bins=b,normed=1,visible=0)[0]*(x*len(serie)/long_len), color = farve, label = navn)
    plt.ylabel('p($\Delta$)')
    plt.xlabel(r'Mismatch power [normalised]')
            
            
    plt.gcf().set_size_inches([ dcolwidth , dcolwidth*0.40])#[9*1.5,3*1.75])
    plt.gcf().set_dpi(400)
    plt.xlim(-1.5,2)
    plt.xticks([-1.5,-1,-0.5,0,0.5,1.0,1.5,2])
    
    handles,labels=ax.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    leg=plt.legend(handles,labels)
    plt.setp(leg.get_texts(),fontsize="small")
    plt.grid(which="major",axis='x')
    plt.tight_layout()
    #~ #show()
    
    plt.savefig('./figures/'+path+'hist_' + title +'.pdf', dpi=300)
    print ("Saved @ ./figures/"+path+"hist_"+title+".pdf")
    plt.close()
    
def linesomatic(S,title, dashed = None, dashed2 = None, y_label = r'$< L >$', x_label = 'Wind/solar mix',x_range=None, y_lim=None, path = '', nolegend = 0):
    plt.close()
    ax = plt.subplot(1,1,1)
    ax.set_xlim(0,1)
    if y_lim: ax.set_ylim(y_lim[0],y_lim[1])
    if x_range == None:
        x_range=np.linspace(0,1,len(S[0][0]))
    
    for s in S:
        (serie, farve, navn) = s
        if navn[0] == 'X':
            ax.plot(x_range, serie, label=navn, color=farve, ls= 'dashed')
        else:
            ax.plot(x_range, serie, label=navn, color=farve)
    if dashed: 
        if len(dashed) == 1: ax.plot([x_range[0],x_range[-1]],[dashed[0]]*2,linestyle="dashed",color='k')
        else: ax.plot(x_range,dashed,linestyle="dashed",color='k')
    if dashed2: 
        if len(dashed2) == 1: ax.plot([x_range[0],x_range[-1]],[dashed2[0]]*2,linestyle="dotted",color='k')
        else: ax.plot(x_range,dashed2,linestyle="dotted",color='k')

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    if not nolegend:
        handles,labels=ax.get_legend_handles_labels()
        Labs=[]
        Hands=[]
        for i in range(len(labels)):
            if labels[i][0] != 'X':
                Labs.append(labels[i])
                Hands.append(handles[i])
        leg=plt.legend(Hands,Labs,loc=0)
        plt.setp(leg.get_texts(),fontsize="small")
    plt.grid(which="major",axis='x')

    plt.gcf().set_size_inches([ 0.5*dcolwidth , 0.5*dcolwidth])
    plt.gcf().set_dpi(400)
    plt.tight_layout()
    plt.savefig("./figures/"+path+"lines_"+title+".pdf")
    print ("Saved @ ./figures/"+path+"lines_"+title+".pdf")

def barmaker(names,S,title="bars", dashed = None, y_label = "Backup capacity",path='',legend=0,y_lim=2,y_range=[0.5,1.0,1.5],stacked = True, dots = None, uglylegend=1):
    #gcf().set_size_inches([15,5])
    width=1.0#0.95
    ind=np.arange(len(S[0][0]))
    plt.close()
    ax = plt.subplot(1,1,1)

    bars=[]
    labels=[]
    ax.set_ylim(0,y_lim)
    ax.set_xlim(-0.875,(len(ind)-1)*1.25+0.875)
    ax.set_yticks(y_range)
    
    if stacked:
        for s in S:
            (serie,farve,navn) = s
            if navn == 'Peak load':
                bars.append(plt.bar(ind*1.25,serie,width=width,align='center',color=farve, ls='dotted'))#, edgecolor="none"))
            else:
                bars.append(plt.bar(ind*1.25,serie,width=width,align='center',color=farve))#, edgecolor="none"))
            labels.append(navn)
        #print mean(alphas[2:])
    else:
        ax.set_xlim(-0.875,(len(ind)-1)*1.25+0.875)
        width /= len(S)*0.8
        offset = np.linspace(-0.15,0.15,len(S))
        for i in range(len(S)):
            
            bars.append(plt.bar(ind*1.25+offset[i],S[i][0],width=width,align='center',color=S[i][1], edgecolor="none"))
            labels.append(S[i][2])
            
    if dashed: ax.plot([-2,50],[dashed]*2,linestyle="dashed",color='k')
    #plt.xticks()
    ax.set_xticks(ind*1.25+.35)
    ax.set_xticklabels(names,rotation = 60, ha="right", va="top")
    ax.set_ylabel(y_label)
    ax.xaxis.grid(False)
    ax.xaxis.set_tick_params(width=0)
    #ax.xaxis.set_ticks([])
    #handles,labels=ax.get_legend_handles_labels()
    #leg=plt.legend(handles,labels)
    #plt.setp(leg.get_texts(),fontsize="small")
    if dots != None:
        for i in range(30):
            plt.plot([ind[i]*1.25-0.1, ind[i]*1.25+0.4],[dots[i],dots[i]],'-',color = orange, lw = 2.0, dash_capstyle='round')
    if uglylegend:
        artists = [plt.Line2D([0,0],[0,0],ls='dashed',lw=2.0,c='k'), plt.Rectangle((0,0),0,0,ec=green,fc=green), plt.Rectangle((0,0),0,0,ec=blue,fc=blue)]
        LABS = ['node proportional M$^1$','link proportional M$^2$','usage proportional M$^3$']
        for b in bars:
            artists.append(b)
        #leg = plt.legend(artists,['link proportional [$M^2$]','usage proportional [$M^3$]'],loc='upper left',ncol=len(bars), columnspacing=0.6,borderpad=0.4, borderaxespad=0.0, handletextpad=0.2, handleheight = 1.2)
        leg = plt.legend(artists, LABS,loc='upper left',ncol=len(artists), columnspacing=0.6,borderpad=0.4, borderaxespad=0.0, handletextpad=0.2, handleheight = 1.2)
        leg.get_frame().set_alpha(0)
        leg.get_frame().set_edgecolor('white')
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=9.5)    # the legend text fontsize
    
    if legend:
        artists = [
        plt.Rectangle((0,0),0,0,fc=red,ec=red),
        plt.Rectangle((0,0),0,0,fc=purple,ec=purple),
        plt.Rectangle((0,0),0,0,fc=darkblue,ec=darkblue),
        plt.Rectangle((0,0),0,0,fc=blue,ec=blue),
        plt.Rectangle((0,0),0,0,fc=grayblue,ec=grayblue),
        plt.Rectangle((0,0),0,0,fc=lightblue,ec=lightblue),
        plt.Rectangle((0,0),0,0,fc=aqua,ec=aqua)]
        leg = plt.legend(artists,labels,loc='upper left',ncol=len(bars), columnspacing=0.6,borderpad=0.4, borderaxespad=0.0, handletextpad=-1.85, handleheight = 1.5)
        leg.get_frame().set_alpha(0)
        leg.get_frame().set_edgecolor('white')
        ltext  = leg.get_texts();
        for i in range(4):
            ltext[i].set_color('w')
        plt.setp(ltext, fontsize=9.5)    # the legend text fontsize

    
    
    
    plt.gcf().set_size_inches([ dcolwidth , dcolwidth*0.4])#[9*1.5,3*1.75])
    plt.gcf().set_dpi(400)
    plt.tight_layout()
    plt.savefig("./figures/"+path+"bars_"+title+".pdf")
    print ("Saved @ ./figures/"+path+"bars_"+title+".pdf")
    plt.close()

def doublebarmaker(names,TOP, BOT, title = "bars", dashed = None, y_label = r'Imports', path=''):
        #gcf().set_size_inches([15,5])
    width=0.33
    ind=np.arange(len(TOP[0][0]))
    plt.close()
    fig = plt.figure()
    ax1 = fig.add_axes([0.09,0.515,0.9,0.425])
    

    bars=[]
    labels=[]
    offset = 0
    for s in TOP:
        (serie,farve,navn) = s
        bars.append(plt.bar(ind*1.25 + offset*0.33 - 0.33,serie,width=width,align='center',color=farve))
        labels.append(navn)
        offset+=1
    #print mean(alphas[2:])
    
    
    ax1.set_ylim(0,0.36)
    ax1.set_xlim(-0.875,(len(ind)-1)*1.25+0.875)
    ax1.set_yticks([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35])
    #plt.xticks()
    ax1.set_xticks(ind*1.25+.35)
    ax1.set_xticklabels(names,rotation = 90, ha="right", va="top")
    ax1.set_ylabel(y_label)
    #handles,labels=ax.get_legend_handles_labels()
    #leg=plt.legend(handles,labels)
    #plt.setp(leg.get_texts(),fontsize="small")
    
    leg = plt.legend(bars,labels,loc='upper left',ncol=len(bars), columnspacing=0.6,borderpad=0.2, borderaxespad=0.4, handletextpad=0.15)


    ax1.tick_params(axis='y',which='both',labelsize=8)
    ax2 = fig.add_axes([0.09,0.025,0.9,0.425])
    ax2.set_ylim(-.05,0.05)
    ax2.set_xlim(-0.875,(len(ind)-1)*1.25+0.875)
    ax2.plot([-10,100],[0,0],linewidth=0.5,color='k')
    ax2.set_ylabel('Net exports')
    ax1.yaxis.set_label_coords(-0.065,0.5)
    ax2.yaxis.set_label_coords(-0.065,0.5)

    offset = 0
    doublevalue=0
    for s in BOT:
        (serie,farve,navn) = s
        plt.bar(ind*1.25 + offset*0.33 - 0.33,serie,width=width,align='center',color=farve, edgecolor="none"*(farve==blue)+"k"*(farve!=blue))
        if farve == blue: doublevalue+=1
        if doublevalue == 1: offset-=1
        offset+=1
        if doublevalue == 2: doublevalue*=0
    ax2.set_xticklabels([])
    #~ ax2.set_yticks([-0.10,-0.05,0.0,0.05,0.1])

    ax2.tick_params(axis='y',which='both',labelsize=8)
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize=9.5)    # the legend text fontsiz
    plt.gcf().set_size_inches([ dcolwidth , dcolwidth*0.7])#[9*1.5,3*1.75])
    plt.gcf().set_dpi(400)
    #plt.tight_layout()
    plt.savefig("./figures/"+path+"doublebars_"+title+".pdf")
    plt.close()
    
def special_barmaker(names,serie,title="bars", y_max = 0.3, y_label = "Backup capacity", path='figures'):
    #gcf().set_size_inches([15,5])
    width=0.95
    ind=np.array([0, 1.25, 2.5, 3.75,4.75,5.75, 7.0, 8.25])
    plt.close()
    ax = plt.subplot(1,1,1)

    bars=[]
    labels=[]
    plt.bar(ind,serie,width=width,align='center',color=blue)
    #print mean(alphas[2:])
    
    ax.set_ylim(0,y_max)
    ax.set_xlim(-0.875,(8.25)+0.875)
    #ax.set_yticks([0.0,0.5,1.0,1.5,2.0])
    ax.set_xticks(ind+.2)
    ax.set_xticklabels(names,rotation = 60, ha="right", va="top")
    ax.set_ylabel(y_label)
    
    plt.gcf().set_size_inches([ dcolwidth , dcolwidth*0.6])#[9*1.5,3*1.75])
    plt.gcf().set_dpi(400)
    plt.tight_layout()
    plt.savefig("./"+path+"/sbars_"+title+".pdf")
    print ("Saved figure ./"+path+"/sbars_"+title+".pdf")
    plt.close()

def special_doublebarmaker(names,TOP, BOT, title = "bars", path = "figures", y_label = r'Average backup power'):
        #gcf().set_size_inches([15,5])
    width=0.95
    ind=np.array([0, 1.25, 2.5, 3.75,4.75,5.75, 7.0, 8.25])
    plt.close()
    fig = plt.figure()
    ax1 = fig.add_axes([0.13,0.57,0.85,0.42])
    
    plt.bar(ind,TOP,width=width,align='center',color=blue)

    #ax1.set_ylim(0,0.36)
    ax1.set_xlim(-0.875,8.25+0.875)
    #ax1.set_yticks([0.0,0.05,0.10,0.15,0.20,0.25,0.30])
    #plt.xticks()
    ax1.set_xticks(ind+.3)
    ax1.set_xticklabels(names,rotation = 90, ha="right", va="top", size = 'xx-small')
    ax1.set_ylabel(y_label,size='small')

    ax1.tick_params(axis='y',which='both',labelsize=8)
    ax2 = fig.add_axes([0.13,0.025,0.85,0.42])
    ax2.set_ylim(-1.20001,0.0)
    ax2.set_xlim(-0.875,8.25+0.875)
    ax2.set_ylabel('Backup capacity',size='small')
    ax1.yaxis.set_label_coords(-0.1,0.5)
    ax2.yaxis.set_label_coords(-0.1,0.5)

    plt.bar(ind,BOT,width=width,align='center',color=blue)
    ax2.set_xticklabels([])
    ax2.set_yticks([-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0])
    ax2.set_yticklabels(('1.2','1.0','0.8','0.6','0.4','0.2','0.0'))
    ax2.tick_params(axis='y',which='both',labelsize=8)
    
    plt.gcf().set_size_inches([ 0.5*dcolwidth , 0.7*dcolwidth])#[9*1.5,3*1.75])
    plt.gcf().set_dpi(400)
    #plt.tight_layout()
    plt.savefig("./"+path+"/special_doublebars_"+title+".pdf")
    plt.close()


def eurograph(N, nvals=np.ones(30), lvals=np.ones(50), title="network", nmode="color", lmode="alpha", n0='DE', l0=('DE','AT'), path='figures'):
    plt.close()
    plt.ioff()
    K,h,ListF=AtoKh_old(N)
    G=nx.Graph()
    nodelist=[str(n.label) for n in N]
    colours=[]
    for n in nodelist:
        G.add_node(n)
    n0_id = [n for n in N if n.label==n0][0].id
    l0_id = [l for l in ListF if (l0[0] in l) and (l0[1] in l)][0][2]

    if nmode == 'color':
        colors = au_cmap(nvals)
    if nmode == 'alpha':
        colors = [au_cmap(1.0)]*len(N)
    ##### Color central node "n0" as red, regardless of other colors
    red_rgb = [int(red[2*i+1:(2*i+1)+2],16)/255.0 for i in range(3)]
    colors[n0_id]=(red_rgb[0],red_rgb[1],red_rgb[2],1)

    w=0
    for l in ListF:
        G.add_edge(l[0], l[1] , weight= lvals[w])
        w+=1

  
    pos=nx.spring_layout(G)


    pos['AT']=[0.55,0.45]
    pos['FI']=[.95,1.1]
    pos['NL']=[0.40,0.85]
    pos['BA']=[0.65,0.15]
    pos['FR']=[0.15,0.60]
    pos['NO']=[0.5,1.1]
    pos['BE']=[0.275,0.775]
    pos['GB']=[0.10,1.05]
    pos['PL']=[0.75,0.8]
    pos['BG']=[0.9,0.0]
    pos['GR']=[0.7,0.0]
    pos['PT']=[0.0,0.15]
    pos['CH']=[0.4,0.45]
    pos['HR']=[0.75,0.3]
    pos['RO']=[1.0,0.15]
    pos['CZ']=[0.75,0.60]
    pos['HU']=[1.0,0.45]
    pos['RS']=[0.85,0.15]
    pos['DE']=[0.45,0.7]
    pos['IE']=[0.0,0.95]
    pos['SE']=[0.75,1.0]
    pos['DK']=[0.5,0.875]
    pos['IT']=[0.4,0.2]
    pos['SI']=[0.55,0.3]
    pos['ES']=[0.15,0.35]
    pos['LU']=[0.325,0.575]
    pos['SK']=[0.90,0.55]
    pos['EE']=[1.0,0.94]
    pos['LV']=[0.95,0.83]
    pos['LT']=[0.87,0.72]
    


    fig = plt.figure(dpi=400,figsize=(1.7*colwidth,1.7*colwidth*0.75))

    ax1= fig.add_axes([-0.125,0.135,1.25,1.0]) #For displaying graph 
    if nmode == 'color':   
        nx.draw_networkx_nodes(G,pos,node_size=600,nodelist=nodelist,node_color=colors,facecolor=(1,1,1))
    if nmode == 'alpha':
        for n in N:
            ncolor = blue
            nalpha = nvals[n.id]
            if str(n.label) == n0: 
                nalpha = 1.0
                ncolor=red
            nx.draw_networkx_nodes(G,pos,node_size=600,nodelist=[str(n.label),],node_color=ncolor,facecolor=(1,1,1),alpha=nalpha)
#    e1=[(u,v) for (u,v,d) in G.edges(data=True)]
#    nx.draw_networkx_edges(G,pos,edgelist=e1,width=1.5,edgecolor=(0.175,0.175,0.175))
    e0=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<=0.1]
    e1=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.1 and d['weight']<=0.2]
    e2=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.2 and d['weight']<=0.3]
    e3=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.3 and d['weight']<=0.4]
    e4=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.4 and d['weight']<=0.5]
    e5=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.5 and d['weight']<=0.6]
    e6=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.6 and d['weight']<=0.7]
    e7=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.7 and d['weight']<=0.8]
    e8=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.9 and d['weight']<=0.9]
    e9=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0.9]
#    ax1.text(-0.05,1.05,"(a)",fontsize=12)
    nx.draw_networkx_edges(G,pos,edgelist=e0,width=1.0,edge_color='k',alpha=1.0,style='dotted')
    nx.draw_networkx_edges(G,pos,edgelist=e1,width=0.5,edge_color='k',alpha=0.7,style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e2,width=1.0,edge_color='k',alpha=0.8,style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e3,width=2.0,edge_color='k',alpha=.9,style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e4,width=3.0,edge_color='k',alpha=1.0,style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e5,width=3.0,edge_color='k',alpha=0.6)
    nx.draw_networkx_edges(G,pos,edgelist=e6,width=3.5,edge_color='k',alpha=.7)
    nx.draw_networkx_edges(G,pos,edgelist=e7,width=4.0,edge_color='k',alpha=.8)
    nx.draw_networkx_edges(G,pos,edgelist=e8,width=4.5,edge_color='k',alpha=0.9)
    nx.draw_networkx_edges(G,pos,edgelist=e9,width=5.0,edge_color='k',alpha=1.0)
    nx.draw_networkx_labels(G,pos,font_size=13,font_color='w',font_family='sans-serif')
    ax1.axis('off') 

#    nx.draw_networkx_labels(G,pos,font_size=8,font_color='k',font_family='sans-serif')

    ax1.axis('off')

    ax4= fig.add_axes([-0.075,0.075,1.5,.15]) #For displaying graph
    ax4.vlines(0.06*1.05+0.025,0.6,1.0,linewidth=1.0,color='k',alpha=1.0,linestyles='dotted')
    ax4.vlines(0.12*1.05+0.025,0.6,1.0,linewidth=0.5,color='k',alpha=0.7,linestyle='dashed')
    ax4.vlines(0.18*1.05+0.025,0.6,1.0,linewidth=1.0,color='k',alpha=0.8,linestyle='dashed')
    ax4.vlines(0.24*1.05+0.025,0.6,1.0,linewidth=2.0,color='k',alpha=0.9,linestyle='dashed')
    ax4.vlines(0.30*1.05+0.025,0.6,1.0,linewidth=3.0,color='k',alpha=1.0,linestyle='dashed')
    ax4.vlines(0.36*1.05+0.025,0.6,1.0,linewidth=3.0,color='k',alpha=0.6)
    ax4.vlines(0.42*1.05+0.025,0.6,1.0,linewidth=3.5,color='k',alpha=0.7)
    ax4.vlines(0.48*1.05+0.025,0.6,1.0,linewidth=4.0,color='k',alpha=0.8)
    ax4.vlines(0.54*1.05+0.025,0.6,1.0,linewidth=4.5,color='k',alpha=0.9)
    ax4.vlines(0.60*1.05+0.025,0.6,1.0,linewidth=5.0,color='k',alpha=1.0)
    ax4.text(0.06*1.05+0.01,0.5,"$\leq$ 10\%",fontsize=9,rotation=-60)
    ax4.text(0.12*1.05+0.01,0.5,"$\leq$ 20\%",fontsize=9,rotation=-60)
    ax4.text(0.18*1.05+0.01,0.5,"$\leq$ 30\%",fontsize=9,rotation=-60)
    ax4.text(0.24*1.05+0.01,0.5,"$\leq$ 40\%",fontsize=9,rotation=-60)
    ax4.text(0.30*1.05+0.01,0.5,"$\leq$ 50\%",fontsize=9,rotation=-60)
    ax4.text(0.36*1.05+0.01,0.5,"$\leq$ 60\%",fontsize=9,rotation=-60)
    ax4.text(0.42*1.05+0.01,0.5,"$\leq$ 70\%",fontsize=9,rotation=-60)
    ax4.text(0.48*1.05+0.01,0.5,"$\leq$ 80\%",fontsize=9,rotation=-60)
    ax4.text(0.54*1.05+0.01,0.5,"$\leq$ 90\%",fontsize=9,rotation=-60)
    ax4.text(0.60*1.05+0.01,0.5,"$\geq$ 90\%",fontsize=9,rotation=-60)
    ax4.axis([0.0,1.0,0.0,1.2])
    ax4.axis('off')

    #plt.tight_layout()
    plt.savefig("./"+path+"/graph_"+title+".pdf")
    #show() # display

def twelve_grid(R=None, title="fullpage", path= 'figures/', singles = 0):
    plt.close()
    fig = plt.figure(dpi=400,figsize=(dcolwidth,1.3*dcolwidth))
    x_range = np.linspace(0,1,21)
    row=0
    for r in range(4):
        for c in range(3):
            ax = fig.add_axes([0.09+(c*0.3),1.0-((r+1)*0.22)-0.05,0.29,0.21])
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.plot(x_range,R[r]['data'][c][0],color=red,label='No transmission')
            plt.plot(x_range,R[r]['data'][c][2],color=orange,label='Synchronised')
            plt.plot(x_range,R[r]['data'][c][1],color=blue,linestyle='dashed', dashes=(30,10),label='Localised')
            if r == 2:
                #plt.plot(x_range,R[r]['data_99'][c][0],color=red, alpha=0.5, lw=1.5)
                #plt.plot(x_range,R[r]['data_99'][c][2],color=orange, alpha=0.5, lw=1.5)
                #plt.plot(x_range,R[r]['data_99'][c][1],color=blue,linestyle='dashed', dashes=(30,10),alpha=0.5, lw=1.5)
                ## Also plot load lines
                #plt.plot([0,1],[1.57169]*2,color='k',alpha=0.7,linestyle='dashed')
                plt.plot([0,1],[1.40189]*2,color='k',alpha=0.4,linestyle='dashed',lw=1.5)
            if c == 0:
                plt.ylabel(R[r]["Y axis"])
                if r == 0:
                    handles,labels=ax.get_legend_handles_labels()
                    handles = [plt.Line2D([0],[0],color=red),plt.Line2D([0],[0],color=blue),plt.Line2D([0],[0],color=orange)]
                    leg=plt.legend(handles,["No transmission", "Localised", "Synchronised"],loc=0)
                    plt.setp(leg.get_texts(),fontsize="small")
            if singles and c == 1:
                plt.close()
                fig = plt.figure(dpi=400,figsize=(colwidth,colwidth))
                ax = fig.add_subplot(1,1,1)  
                plt.tick_params(axis='both', which='major', labelsize=8)
                plt.plot(x_range,R[r]['data'][c][0],color=red,label='No transmission')
                plt.plot(x_range,R[r]['data'][c][2],color=orange,label='Synchronised')
                plt.plot(x_range,R[r]['data'][c][1],color=blue,linestyle='dashed', dashes=(30,10),label='Localised') 
                plt.xlabel("Wind/solar mix")
                plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0]) 
                plt.ylim = R[r]['Y range']
                plt.yticks(R[r]['Y ticks'])
                plt.ylabel(R[r]["Y axis"])
                handles,labels=ax.get_legend_handles_labels()
                leg=plt.legend(handles,labels,loc=0)
                plt.setp(leg.get_texts(),fontsize="small")
                plt.tight_layout()
                plt.savefig("./"+path+"twelvegrid_"+title+str(r)+".pdf")
            
            if r == 0:
                plt.xlabel(R[r]["X axis"][c])
                ax.xaxis.set_label_position('top')
                plt.xticks([])
            elif r!=3:
                plt.xticks([])
            else:
                plt.xlabel("Wind/solar mix")
                plt.xticks([0.2,0.4,0.6,0.8])
            plt.ylim = R[r]['Y range']
            plt.yticks(R[r]['Y ticks'])
            if c == 0:
                plt.setp(ax.get_yticklabels(), visible=True)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim(R[r]['Y range'])
    plt.savefig("./"+path+"/twelvegrid_"+title+".pdf")
    
def subplot_fillbetween(ax,tc,wc,sc,bc,be,ee=None,shade=None):
    As = np.linspace(0,1,101)
    a_00 = tc
    a_01 = tc + wc
    a_02 = tc + wc + sc
    a_03 = tc + wc + sc +bc
    a_04 = tc + wc + sc +bc + be
    if ee != None:
        a_0B = -ee
        #a_00 = -ee + tc
        #a_01 = -ee + tc + wc
        #a_02 = -ee + tc + wc + sc
        #a_03 = -ee + tc + wc + sc +bc
        b_04 = -ee + tc + wc + sc +bc + be
    
    a_opt = As[np.where(a_04 == min(a_04))[0][0]]

    a_o = int(100*a_opt)
    print "For no transmisison case, optimal alpha is ",a_opt, "and costs ",min(a_04)," at gamma = 1"

    ax.fill_between(As,a_04,facecolor=orange)
    ax.fill_between(As,a_03,facecolor=red)
    ax.fill_between(As,a_02,facecolor=yellow)
    ax.fill_between(As,a_01,facecolor=blue)
    ax.fill_between(As,a_00,facecolor=green)
    if ee!= None: 
        ax.fill_between(As,a_0B,facecolor=aqua)
        ax.plot(As,b_04,linestyle='dashed', color='k')
        print "real minimum at ", min(b_04)

    ax.set_xlim(0,1)
    ax.set_ylim(-2,14)
    ['$\gamma = 0.5$','$\gamma = 1.0$', '$\gamma = 1.5$']
    ax.plot(a_opt,min(a_04),'o',color='k')
    ax.plot([a_opt,a_opt],[0,min(a_04)],linestyle="dashed",color='k')
    if shade:
        ax.plot(shade[0],shade[1],'o',color='k',alpha=0.4)
        ax.plot(As,shade[2],linestyle="dashed",color='k',alpha=0.4)
    return [a_opt,min(a_04),a_04]
    
def subplot_colormap(ax,wc=1.0,be=1.0, pos = '', notran = False):
    Gs = np.linspace(0,2,101)
    As = np.linspace(0,1,101)


    Cs = (np.load("matrices/SYN_B_C.npy") + np.load("matrices/SYN_B_E.npy")*be + np.load("matrices/SYN_W_C.npy")*wc + np.load("matrices/SYN_S_C.npy") + np.load("matrices/SYN_T_C.npy"))/Energy()
    if notran:
        Cs = np.load("sensitivity_results/N_B_C99.npy") + np.load("sensitivity_results/N_B_E.npy")*be/8.0 + np.load("sensitivity_results/N_W_C.npy")*wc + np.load("sensitivity_results/N_S_C.npy")
    
    #im = plt.imshow(Cs,cmap = plt.cm.coolwarm, vmin=2.7, vmax=8.1)#plt.cm.coolwarm
    #contour_range = [3.2,3.5,3.8,4.2,4.6,5.1,5.7,6.5,7.8,9.1]
    im = plt.imshow(Cs,cmap = plt.cm.coolwarm, vmin=48, vmax=179)#plt.cm.coolwarm
    contour_range = [76,84,90,100,110,120,135,155,185,210]
    contour_range = [56,60,66,72,78,80,90,100,120,140]
    if pos == 'top left':
            contour_range = [64,66,72,80,90,100,120,140]
    if pos == 'top right':
            contour_range = [74,76,80,86,95,115,130,160]
    if pos == 'bot left':
            contour_range = [48,52,58,62,70,80,100,120,140]
    if pos == 'bot right':
            contour_range = [53,58,64,70,80,90,100,120,140]
    contour = plt.contour(Cs,contour_range,linewidths=2.0,colors='k')#cmap=plt.cm.coolwarm_r)#,interpolation='none')
    plt.clabel(contour,inline=True,fmt='%1.f',fontsize=12)
    plt.grid('on')
    plt.xlabel(r'VRES mix ($\alpha_W$)')
    plt.xlim(1,100)
    plt.ylim(1,100)
    plt.xticks([20,40,60,80],[0.2,0.4,0.6,0.8])
    plt.yticks([20,40,60,80],[0.4,0.8,1.2,1.6])
    Ks = 1.0*np.array(Cs)
    print Ks.min(), Ks.max()
    minimu = np.where(Ks == Ks.min())
    x = minimu[0][0]
    y = minimu[1][0]
    print minimu,"minima at alpha ",(y)/(101.),"gamma",x/(101.)*2
    plt.plot(y,x,'ko')
    plt.ylabel(r'VRES penetration ($\gamma$)')
    plt.tick_params(axis='both', which='major')#, labelsize=8)
    if 'top' in pos:
        plt.setp(ax.get_xticklabels(), visible  = False)
        plt.xlabel('')
    if 'right' in pos:
        plt.ylabel('')
        plt.setp(ax.get_yticklabels(), visible  = False)
    if pos == 'top left': plt.text(2,3,'(a)', fontsize='18')
    if pos in ['top right','mid top','bot of two']: plt.text(2,3,'(b)', fontsize='18')
    if pos in ['bot left','bot of three']: plt.text(2,3,'(c)', fontsize='18')
    if pos == 'bot right': plt.text(2,3,'(d)', fontsize='18')
    
    return im
    

def six_grid(L=None, M=None,R =None, title="fullpage", path= 'figures/', Ylabels=['$\gamma = 0.5 \qquad$ \n System cost [TE]','$\gamma = 1.0 \qquad$ \n System cost [TE]', '$\gamma = 1.5\qquad$ \n System cost [TE]'], Xlabels = ['No transmission','Localised','Synchronised'],singles=0):
    #old Ylabels=[]
    plt.close()
    fig = plt.figure(dpi=400,figsize=(dcolwidth,1.2*dcolwidth))
    x_range = np.linspace(0,1,21)
    row=0
    shade = None
    G = [L,M,R]
    ylims=[5.5,6.5,7.5]
    yticos=[[1,2,3,4,5],[1,2,3,4,5,6],[1,2,3,4,5,6,7]]
    ylims=[120,120,120]#[80,110,130]
    yticos=[[10,30,50,70,],[20,40,60,80,100,],[20,40,60,80,100,120]]

    
    for r in range(3):
        for c in range(3):
            ax = fig.add_axes([0.09+(c*0.3),1.0-((r+1)*0.3)-0.05,0.29,0.29])
            plt.tick_params(axis='both', which='major', labelsize=8)
            if singles and r == 1:
                plt.close()
                fig = plt.figure(dpi=400,figsize=(1.2*colwidth,1.2*colwidth))
                ax = fig.add_subplot(1,1,1)
                subplot_fillbetween(ax,tc=G[c][r]['tc'],wc=G[c][r]['wc'],sc=G[c][r]['sc'],bc=G[c][r]['bc'],be=G[c][r]['be'])
                plt.ylabel( 'LCOE ['+EUR+'/MWh]')
                if c == 0:
                    plt.text(0.58,95,Xlabels[c])
                if c == 1:
                    plt.text(0.7,95,Xlabels[c])
                if c == 2:
                    plt.text(0.63,95,Xlabels[c])
                plt.xlabel("Wind/solar mix")
                plt.xticks([0.2,0.4,0.6,0.8])
                
                artists = [plt.Rectangle((0,0),0,0,fc=orange),plt.Rectangle((0,0),0,0,fc=red),plt.Rectangle((0,0),0,0,fc=yellow),plt.Rectangle((0,0),0,0,fc=blue),plt.Rectangle((0,0),0,0,fc=green)]
                l_names = ['Backup energy', 'Backup capacity', 'Solar capacity', 'Wind capacity', 'Transmission capacity']
                leg = plt.legend(artists,l_names,loc='upper right',ncol=2, prop={'size':8}, handletextpad=0.15, columnspacing = 1.0);
                ax.set_ylim(0,120)#ylims[r])
                plt.yticks([0,20,40,60,80,100,120])#yticos[r])
                plt.tight_layout()
                plt.savefig("./"+path+"sixgrid_"+title+str(c)+".pdf")

                
            
            if c == 0:
                shade = subplot_fillbetween(ax,tc=G[c][r]['tc'],wc=G[c][r]['wc'],sc=G[c][r]['sc'],bc=G[c][r]['bc'],be=G[c][r]['be'],shade=shade)
            else:
                subplot_fillbetween(ax,tc=G[c][r]['tc'],wc=G[c][r]['wc'],sc=G[c][r]['sc'],bc=G[c][r]['bc'],be=G[c][r]['be'],shade=shade)
            if c == 0:
                plt.ylabel(Ylabels[r])
            if c == 2:
                shade = None
            if r == 0:
                plt.xlabel(Xlabels[c])
                ax.xaxis.set_label_position('top')
                plt.xticks([])
            elif r!=2:
                plt.xticks([])
            else:
                plt.xlabel("Wind/solar mix")
                plt.xticks([0.2,0.4,0.6,0.8])
            if c == 0:
                plt.setp(ax.get_yticklabels(), visible=True)
                if r == 2:
                        artists = [plt.Rectangle((0,0),0,0,fc=orange),plt.Rectangle((0,0),0,0,fc=red),plt.Rectangle((0,0),0,0,fc=yellow),plt.Rectangle((0,0),0,0,fc=blue),plt.Rectangle((0,0),0,0,fc=green)]
                        l_names = ['Backup energy', 'Backup capacity', 'Solar capacity', 'Wind capacity', 'Transmission capacity']
                        leg = plt.legend(artists,l_names,loc='upper center',ncol=len(artists), bbox_to_anchor=(0.535,0.94), bbox_transform = plt.gcf().transFigure, prop={'size':8}, handletextpad=0.15, columnspacing = 1.0);
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim(0,120)#ylims[r])
            plt.yticks([20,40,60,80,100])#yticos[r])
    plt.savefig("./"+path+"/sixgrid_"+title+".pdf")
    


def three_grid(G, title="fullpage", path= 'figures', Ylabels=['$\gamma = 0.5$; \quad System cost [TE]','$\gamma = 1.0$; \quad System cost [TE]', '$\gamma = 1.5$; \quad System cost [TE]'], Xlabels = ['No transmission','']):
    plt.close()
    fig = plt.figure(dpi=400,figsize=(colwidth,2.3*colwidth))
    x_range = np.linspace(0,1,21)
    row=0
    shade = None
    ylims=[90,110,130]
    yticos=[[1,2,3,4],[1,2,3,4,5],[1,2,3,4,5,6]]
    for r in range(3):
            ax = fig.add_axes([0.18,1.0-((r+1)*0.31)-0.01,0.8,0.3])
            plt.tick_params(axis='both', which='major', labelsize=8)
            a_04 = G[r]['tc'] + G[r]['wc'] + G[r]['sc'] + G[r]['bc'] + G[r]['be']
            shade = [np.linspace(0,1,101)[np.where(a_04 == min(a_04))[0][0]], min(a_04),a_04]
            print "minimum at ", a_04
            s = subplot_fillbetween(ax,tc=G[r]['tc'],wc=G[r]['wc'],sc=G[r]['sc'],bc=G[r]['bc'],be=G[r]['be'],ee = G[r]['ee'], shade=shade)
            plt.ylabel(Ylabels[r])
            if r == 0:
                plt.xlabel(Xlabels[1])
                ax.xaxis.set_label_position('top')
                plt.xticks([])
                
                artists = [plt.Rectangle((0,0),0,0,fc=orange),plt.Rectangle((0,0),0,0,fc=red),plt.Rectangle((0,0),0,0,fc=yellow),plt.Rectangle((0,0),0,0,fc=blue),plt.Rectangle((0,0),0,0,fc=green), plt.Rectangle((0,0),0,0,fc=aqua)]
                l_names = ['Backup energy', 'Backup capacity', 'Solar capacity', 'Wind capacity', 'Transmission', 'Excess energy']
                leg = plt.legend(artists,l_names,loc='upper center',ncol=len(artists)/3, bbox_to_anchor=(0.63,0.985), bbox_transform = plt.gcf().transFigure, prop={'size':8}, handletextpad=0.15, columnspacing = 1.0);
            elif r!=2:
                plt.xticks([])
            else:
                plt.xlabel("Wind/solar mix")
                plt.xticks([0.2,0.4,0.6,0.8])
            ax.set_ylim(-40,130)
            plt.yticks([-20,0,20,40,60,80,100,120])
    plt.savefig("./"+path+"/threegrid_"+title+".pdf")
    


def fourgrid():    ############  Figure making!
    plt.clf()
    plt.ioff()
    fig = plt.figure()
    subplot_colormap(plt.axes([0.08,0.54,0.45,0.48]), pos = 'top left')
    subplot_colormap(plt.axes([0.54,0.54,0.45,0.48]),be=1.5, pos = 'top right')
    subplot_colormap(plt.axes([0.54,0.12,0.45,0.48]),wc=0.5,be=1.5, pos = 'bot right')
    im = subplot_colormap(plt.axes([0.08,0.12,0.45,0.48]),wc=0.5, pos = 'bot left')
    
    cax = plt.axes([0.08,0.06,0.91,0.04])
    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    plt.tick_params(axis='both', which='major')#, labelsize=8)
    cbar.set_label('LCOE ['+EUR+'/MWh]')
    plt.gcf().set_size_inches([ dcolwidth , 1.1*dcolwidth])#[9*1.5,3*1.75])
    #plt.tight_layout()
    plt.savefig('./figures/colourfour.pdf')#'+costs+'_cost.pdf')
    

def colourtwo():    ############  Figure making!
    plt.clf()
    plt.ioff()
    fig = plt.figure()
    subplot_colormap(plt.axes([0.18,0.54,0.8,0.48]), notran = True, pos = 'top left')
    im = subplot_colormap(plt.axes([0.18,0.12,0.8,0.48]),notran = False, pos = 'bot of two')
    
    cax = plt.axes([0.18,0.06,0.8,0.03])
    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    plt.tick_params(axis='both', which='major')#, labelsize=8)
    cbar.set_label(r'System cost [TE]')
    plt.gcf().set_size_inches([ colwidth , 2.0*colwidth])#[9*1.5,3*1.75])
    #plt.tight_layout()
    plt.savefig('./SENSITIVITY/colourtwo.pdf')#'+costs+'_cost.pdf')
    
def colourthree():
    plt.clf()
    plt.ioff()
    fig = plt.figure()
    
    subplot_colormap(plt.axes([0.18,0.705,0.8,0.3]),be=1.5, pos = 'top left')
    subplot_colormap(plt.axes([0.18,0.405,0.8,0.3]),wc=0.5, pos = 'mid top')
    im = subplot_colormap(plt.axes([0.18,0.105,0.8,0.3]),wc=0.5, be=1.5, pos = 'bot of three')
    
    cax = plt.axes([0.18,0.045,0.8,0.02])
    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    plt.tick_params(axis='both', which='major')#, labelsize=8)
    cbar.set_label(r'System cost [TE]')
    plt.gcf().set_size_inches([ colwidth , 2.8*colwidth])#[9*1.5,3*1.75])
    #plt.tight_layout()
    plt.savefig('./SENSITIVITY/colourthree.pdf')#'+costs+'_cost.pdf')


def drawnet(N=None,F=None,t=0,mode="instant",title="",path=''): ## All the network figures
    plt.close()
    if N==None:
        N=EU_Nodes()
    G=nx.Graph()
    nodelist=[]

    for n in N:
        G.add_node(str(n.label))
        nodelist.append(str(n.label))

    K,h,LF=AtoKh_old(N)

    if 'capacities' in mode:
        w = [link_q(f) for f in F[:,t]]

    if 'instant' in mode:
        w = F[:,t]
        
    for l in LF:
        G.add_edge(l[0], l[1] , weight= np.abs(w[l[2]]))


    pos={}
    pos['AT']=[0.55,0.45]
    pos['FI']=[.95,1.1]
    pos['NL']=[0.40,0.85]
    pos['BA']=[0.65,0.15]
    pos['FR']=[0.15,0.60]
    pos['NO']=[0.5,1.1]
    pos['BE']=[0.275,0.775]
    pos['GB']=[0.10,1.05]
    pos['PL']=[0.75,0.8]
    pos['BG']=[0.9,0.0]
    pos['GR']=[0.7,0.0]
    pos['PT']=[0.0,0.15]
    pos['CH']=[0.4,0.45]
    pos['HR']=[0.75,0.3]
    pos['RO']=[1.0,0.15]
    pos['CZ']=[0.75,0.60]
    pos['HU']=[1.0,0.45]
    pos['RS']=[0.85,0.15]
    pos['DE']=[0.45,0.7]
    pos['IE']=[0.0,0.95]
    pos['SE']=[0.75,1.0]
    pos['DK']=[0.5,0.875]
    pos['IT']=[0.4,0.2]
    pos['SI']=[0.55,0.3]
    pos['ES']=[0.15,0.35]
    pos['LU']=[0.325,0.575]
    pos['SK']=[0.90,0.55]
    pos['EE']=[1.0,0.985]
    pos['LV']=[0.975,0.87]
    pos['LT']=[0.925,0.77]

    fig = plt.figure(dpi=400,figsize=(0.85*dcolwidth,0.85*0.8*dcolwidth))

    #ax1 = fig.add_axes([0.875,0.05,0.05,0.92])
    ## coor bar in middle
    ax1 = fig.add_axes([0.05,0.08,0.9,.08])
    
    au_cdict = {'red': (
    (0.0,int(red[1:3],16)/255.0,int(red[1:3],16)/255.0),
    (0.5,int(darkblue[1:3],16)/255.0,int(darkblue[1:3],16)/255.0),
    (0.75,int(blue[1:3],16)/255.0,int(blue[1:3],16)/255.0),
    (1.0,int(lightblue[1:3],16)/255.0,int(lightblue[1:3],16)/255.0)),
    
    'green': (
    (0.0,int(red[3:5],16)/255.0,int(red[3:5],16)/255.0),
    (0.5,int(darkblue[3:5],16)/255.0,int(darkblue[3:5],16)/255.0),
    (0.75,int(blue[3:5],16)/255.0,int(blue[3:5],16)/255.0),
    (1.0,int(lightblue[3:5],16)/255.0,int(lightblue[3:5],16)/255.0)),
    
    'blue': (
    (0.0,int(red[5:7],16)/255.0,int(red[5:7],16)/255.0),
    (0.5,int(darkblue[5:7],16)/255.0,int(darkblue[5:7],16)/255.0),
    (0.75,int(blue[5:7],16)/255.0,int(blue[5:7],16)/255.0),
    (1.0,int(lightblue[5:7],16)/255.0,int(lightblue[5:7],16)/255.0))}

    cmap = LinearSegmentedColormap('au_cmap',au_cdict,256)
    #cmap = mpl.cm.RdYlGn## rolando: RdYlGn   /// anders: anderscdict  
    
    norm = mpl.colors.Normalize(vmin=-1,vmax=1)
    cbl = mpl.colorbar.ColorbarBase(ax1,cmap,norm,orientation='vhorizontal')
    ax1.set_xlabel(r"Mismatch $\left <L \right >$")
    ax1.xaxis.set_label_position('top') 
    
    #ax2 = fig.add_axes([-0.075,-0.075,0.975,1.2])
    
    ax2 = fig.add_axes([-0.05,0.15,1.1,0.95])
    for l in LF:
        if -200<w[l[2]]<200: continue
        if w[l[2]]>=200:
            x0 = pos[l[0]][0]
            y0 = pos[l[0]][1]
            x1 = pos[l[1]][0]
            y1 = pos[l[1]][1]
        if w[l[2]]<=-200:
            x1 = pos[l[0]][0]
            y1 = pos[l[0]][1]
            x0 = pos[l[1]][0]
            y0 = pos[l[1]][1]
        dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
        ax2.arrow(x0, y0, 0.4*(x1-x0), 0.4*(y1-y0), fc=blue, ec=blue,head_width=min(dist*0.15,dist*0.2*(abs(w[l[2]]))/3000.0 + 0.01), head_length = 0.15*dist)
        
    node_c=[ (cmap(0.5+0.5*n.mismatch[t]/n.mean)) for n in N]
    
    darknodes=[[],[]]
    for i in range(len(N)):
        if np.abs(N[i].mismatch[t]/N[i].mean)<0.4:
            darknodes[0].append(nodelist[i])
            darknodes[1].append(node_c[i])
    nx.draw_networkx_nodes(G,pos,node_size=600,nodelist=nodelist,node_color=node_c,facecolor=(1,1,1))
    
    e10=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<=700000]
    e1=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>700 and d['weight']<=1200]
    e2=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>1200 and d['weight']<=1800]
    e3=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>1800 and d['weight']<=2400]
    e4=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>2400 and d['weight']<=3300]
    e5=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>3300 and d['weight']<=4000]
    e6=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>4000 and d['weight']<=5500]
    e7=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>5500 and d['weight']<=8000]
    e8=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>8000 and d['weight']<=12000]
    e9=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>12000]
#    ax1.text(-0.05,1.05,"(a)",fontsize=12)
    #~ nx.draw_networkx_edges(G,pos,edgelist=e0,width=1.0,edge_color=blue,alpha=1.0,style='dotted')
    #~ nx.draw_networkx_edges(G,pos,edgelist=e1,width=0.5,edge_color=blue,alpha=0.7,style='dashed')
    #~ nx.draw_networkx_edges(G,pos,edgelist=e2,width=1.0,edge_color=blue,alpha=0.8,style='dashed')
    #~ nx.draw_networkx_edges(G,pos,edgelist=e3,width=2.0,edge_color=blue,alpha=.9,style='dashed')
    #~ nx.draw_networkx_edges(G,pos,edgelist=e4,width=3.0,edge_color=blue,alpha=1.0,style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e10,width=3.0,edge_color=blue,alpha=0.6)
    #~ nx.draw_networkx_edges(G,pos,edgelist=e6,width=3.5,edge_color=blue,alpha=.7)
    #~ nx.draw_networkx_edges(G,pos,edgelist=e7,width=4.0,edge_color=blue,alpha=.8)
    #~ nx.draw_networkx_edges(G,pos,edgelist=e8,width=4.5,edge_color=blue,alpha=0.9)
    #~ nx.draw_networkx_edges(G,pos,edgelist=e9,width=5.0,edge_color=blue,alpha=1.0)
    nx.draw_networkx_labels(G,pos,font_size=13,font_color='w',font_family='sans-serif')
    ax2.axis('off') 


    #plt.tight_layout()
    plt.savefig("./figures/"+path+"network_"+title+".pdf")
    #show() # display
