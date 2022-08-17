Hier findest du in kurzer Form:

- Was ist ein (Jupyter-)Notebook?
- Hinweise zur Installation
- Bemerkungen zu den Zelltypen
- Anmerkungen zu Besonderheiten im Umgang mit Jupyter-Notebooks

# Was ist ein *Notebook?*

Ein **Notebook** ist (nach Wikipedia) ein Notizbuch, das für viele Zwecke verwendet wird, z.B. zum Aufzeichnen von Notizen, zum Schreiben von Texten oder zum Darstellen von Graphiken.

Ein Notizbuch wird verwendet, um eine Idee oder eine wichtige Information mithilfe von Texten, Diagrammen, Zeichnungen, Bildern, Gleichungen, Tabellen auszudrücken.

# Was ist ein *Jupyter-Notebook?*

Ein **Jupyter-Notebook** ist in diesem Sinne tatsächlich ein Notizbuch; denn es wird benutzt, um Texte, Programm-Codes, Bilder, Gleichungen, Diagramme, Tabellen, Visualisierungen zu erzeugen und darzustellen.
        
**Jupyter Notebook** ist eine Open-Source-Webanwendung, mit der man Dokumente erstellen und freigeben kann. Es enthält Programm-Code, Visualisierungen Bilder, Text...
        
Um Jupyter-Notebooks benutzen zu können, ist ein Jupyter-System notwendig.

# Das *Jupyter-System*

***Project Jupyter*** ist eine Non-Profit-Organisation, die gegründet wurde, um *Open-Source-Software, offene Standards und Services für interaktives Arbeiten mit Dutzenden Programmiersprachen zu entwickeln*.
        
Das Jupyter-System unterstützt über 100 Programmiersprachen (im Jupyter-System als ***Kernel*** bezeichnet), darunter *Python*, *Java*, *R*, *Julia*, *Matlab* und viele mehr.

Am häufigsten wird das Jupyter-Notizbuch mit einem Python-Kernel verwendet.

# Externes *Jupyter-System* nutzen

Um Jupyter-Notebooks zu benutzen, können Sie den folgenden Link nutzen:

[Jupyter-Server](https://unterrichtsmaterial-ddi.cs.uni-paderborn.de/jhub/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fklausbovermann%2FJupyNBs&urlpath=tree%2FJupyNBs%2Findex.ipynb&branch=main)

Sie können dabei temporäre Kopien von Notebooks sehen und bearbeiten.

# *Anaconda* als lokales Jupyter-System

Um Jupyter-Notebooks benutzen zu können, kann man z.B. ***Anaconda*** installieren.
        
Auf der Seite _[Anaconda](https://www.anaconda.com/ )_ findet man die zu Ihrem Betriebssystem passende Version und eine Installationsanweisung.
    
Nach erfolgreicher Installation kann man den Jupyter-Notebook-Server starten, so dass eigene oder fremde Notebooks geladen und neue Notebooks erzeugt werden können.

# Zellen in einem Jupyter-Notebook

Jedes Jupyter-Notebook besteht aus Zellen. Es gibt zwei (für diesen Zusammenhang interessierende) Zelltypen:

- Code-Zellen. Man erkennt sie an `In [nummer]:` am linken Rand neben der Zelle.
  - In einer solchen Zelle kann man Python-Code eingeben und ausführen lassen. Mehr dazu später.
- Text-Zellen.
  - In diese Zellen kann man Texte, Bilder, Tabellen, Links, Aufzählungen, ... aufnehmen. Dazu benutzt man die Auszeichnungssprache ***MarkDown***, eine Art *Simple-HTML*. Bitte informieren Sie sich bei Bedarf im Internet, z.B. auf den Seiten [MarkDown](https://markdown.de/).

# Python-Programme in Notebooks

Ein Python-Programm beinhaltet in der Regel alle Definitionen, die notwendig sind, sowie den Code des *Hauptprogramms*. Externe Funktionen können über Importe eingebunden werden.

In einem Jupyter-Notebook sucht man (zunächst) vergebens nach einem Hauptprogramm. Man findet stattdessen (neben Zellen, in denen Texte oder auch Bilder zu finden sind) viele **Eingabezellen**, in denen (einzeiliger oder mehrzeiliger) Python-Code steht.

Jede Eingabezelle ist (sofern syntaktisch korrekt) eine ausführbare Python-Anweisung. Also kann man jede Eingabezelle manuell ausführen (Cursor in die Zelle bewegen und per <Shift><Enter> abschließen), und die Reaktion des Systems beobachten. Falls keine Fehlermeldung erfolgt, wurde die Python-Anweisung ausgeführt:
    
- Falls die Zelle eine Import-Anweisung enthalten hat, wurde diese Bibliotheke importiert. Dann kann man in anderen Zellen die Funktionen aus dieser Bibliothek nutzen. Der Kernel dieses Notebooks hat diese Bibliothek integriert.
- Falls in der Zelle eine Variablenzuweisung erfolgte, wird diese Variablenbelegung dauerhaft in dem Kernel aufbewahrt. Diese Variable existiert dann weiterhin ab jetzt in allen anderen Zellen. 
    
Dieses Verhalten ist zunächst ungewohnt, da die Sichtbarkeit einer Variablen oder einer importierten Bibliotheksfunktion dauerhaft in dem aktiven Kernel dieses Notebooks ist, also auch in Zellen *oberhalb* derjenigen, in der die Aktivierung erfolgte.

Falls man also die folgende Zelle aktiviert, erfolgt zunächst eine Fehlermeldung (entfernen Sie zuvor die Kommentar-Marke):


```python
# print (x)
```

Jetzt aktivieren Sie die folgende Zelle:


```python
x = 42
print (x)
```

    42


Wenn Sie jetzt die erstgenannte Zelle erneut aktivieren, erfolgt **keine(!)** Fehlermeldung mehr.

Starten Sie den Kernel erneut, ist die Variablenbindung von `x` aufgehoben.

**Tipp**:

- Beim Erstellen eines Notebooks sollte man die Reihenfolge der Zellen so vornehmen, dass ein sequenzieller Durchlauf zum erwünschten Ergebnis führt.
- Als Benutzer eins (fremden) Notebooks ist es ratsam, die Zellen sequenziell zu aktivieren. Falls man Änderungen vorgenommen hat (z.B. Variablen neu belegt), kann man den Kernel erneut initialisieren und starten (auf Wunsch mit oder ohne automatische Zellenaktivierungen).

***Jetzt wünsche ich Ihnen viel Spaß beim Arbeiten mit Python und Jupyter-Notebooks.***


```python

```
