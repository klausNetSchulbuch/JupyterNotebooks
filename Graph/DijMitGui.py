#!/usr/bin/env python
# coding: utf-8

# # Dijkstra mit GUI

# ## Importieren von Bibliotheken

# In[ ]:


from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
from tkinter import filedialog


import nrw_graph_unsave as ng
import pandas as pd
# import networkx as nx


# ## Datei einlesen

# In[ ]:


def dieKanten(datname):
    return pd.read_csv(datname, sep=",")


# ## Graph (neu) bauen

# In[ ]:


def derGraph():
    autobahn = ng.nrw_graph()
    
    datname = filedialog.askopenfilename(filetypes=[('Graph txt-Files','.txt')])

#    df_kanten = dieKanten("Highway_Germany.txt")
    df_kanten = dieKanten(datname)
    zeilen = df_kanten.shape[0]
    for i in range(zeilen):
        source = df_kanten.iloc[i]['Start']
        target = df_kanten.iloc[i]['Ziel']
        dist = float(df_kanten.iloc[i]['Entfernung'])

        autobahn.fuegeKanteHinzu(source, target, gewicht=dist)
        autobahn.deflagKnoten(source)
        autobahn.deflagKnoten(target)
    
    return autobahn


# In[ ]:


def graphDemarkAll(g):
    for knoten in g.alleKnoten():
        g.deflagKnoten(knoten)


# ## Hilfsfunktionen für den Algorithmus

# In[ ]:


def entfernung (liste):
    return liste[2]


# In[ ]:


def alleSchnittkanten(meinGraph):
    kanten = []
    for (s,z) in meinGraph.alleKanten():
        if meinGraph.knotenHatFlag(s) and not meinGraph.knotenHatFlag(z):
            l = meinGraph.kantenGewicht(s, z)
            (ueber, weit) = meinGraph.getKnotenMarke(s)
            l += weit
            kanten.append ([s, z, l])
        elif meinGraph.knotenHatFlag(z) and not meinGraph.knotenHatFlag(s):
            l = meinGraph.kantenGewicht(z, s)
            (ueber, weit) = meinGraph.getKnotenMarke(z)
            l += weit
            kanten.append([z, s, l])

    kanten.sort(key = entfernung)
    return kanten


# In[ ]:


def alleBesuchtenKnoten(meinGraph):
    alle = []
    for knoten in meinGraph.alleKnoten():
        if meinGraph.knotenHatFlag(knoten):
            (ueber, lang) = meinGraph.getKnotenMarke(knoten)
            alle.append([knoten,ueber, lang])
    return alle


# ## Der Algorithmus

# In[ ]:


def dij (g, startknoten, zielknoten):
    graphDemarkAll(g)
    g.flagKnoten(startknoten)
    g.markiereKnoten(startknoten, (startknoten, 0.0))
    
    while not g.knotenHatFlag(zielknoten):
        sk = alleSchnittkanten(g)
        shortest = sk [0]
        ueber = shortest[0]
        neu = shortest[1]
        lang = shortest[2]
        g.flagKnoten(neu)
        g.markiereKnoten(neu, (ueber, lang))
    
    besucht = {}
    for kante in alleBesuchtenKnoten(g):
        besucht[kante[0]] = [kante[1], kante[2]]

    node = zielknoten
    ausgabe = ""
    gesamt = besucht[zielknoten][1]
    while node != startknoten:
        info = besucht[node]
        ueber = info[0]
        lang = g.kantenGewicht(ueber, node)
        ausgabe = "Von " + ueber + " nach " + node + " (" + str(lang) + ")" + "\n" + ausgabe
        node = ueber
    ausgabe = "Es sind insgesamt " + str(gesamt) + "km" + "\n" + ausgabe  
    return ausgabe


# ## Die Anwendung

# In[ ]:


meinGraph = derGraph()

fenster = Tk()
fenster.title("Dijkstra")

def act(g, startknoten, zielknoten):
    
    text_area.delete('0.0', END)
    out = dij(g, startknoten, zielknoten)
    text_area.insert(INSERT,out)

labelStart = Label(fenster, text = "Startknoten wählen")
labelStart.pack(padx=5, pady=5)

selected_Start = StringVar()
start_cb = ttk.Combobox(fenster, textvariable=selected_Start)
start_cb['state'] = 'readonly'
start_cb.pack(padx=5, pady=5)
start_cb['values'] = sorted(meinGraph.alleKnoten())

labelZiel = Label(fenster, text = "Zielknoten wählen")
labelZiel.pack(padx=5, pady=5)

selected_Ziel = StringVar()
ziel_cb = ttk.Combobox(fenster, textvariable=selected_Ziel)
ziel_cb['values'] = sorted(meinGraph.alleKnoten())
ziel_cb['state'] = 'readonly'
ziel_cb.pack(padx=5, pady=5)

dijkstraButton = Button(fenster, 
                        text = "Search", 
                        command = lambda: act(meinGraph,
                                              selected_Start.get(), 
                                              selected_Ziel.get())
                       )

dijkstraButton.pack(padx = 10, pady = 10)

text_area = scrolledtext.ScrolledText(fenster, 
                                      wrap = WORD, 
                                      width = 100, 
                                      height = 100, 
                                      font = ("Times New Roman", 15))
  
text_area.pack(pady = 50, padx = 50)
                 
def start_changed(event):
    showinfo(
        title='Start',
        message=f'You selected {selected_Start.get()} as Start!'
    )

def ziel_changed(event):
    showinfo(
        title='Ziel',
        message=f'You selected {selected_Ziel.get()} as Ziel!'
    )
    
start_cb.bind('<<ComboboxSelected>>', start_changed)
ziel_cb.bind('<<ComboboxSelected>>', ziel_changed)


fenster.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




