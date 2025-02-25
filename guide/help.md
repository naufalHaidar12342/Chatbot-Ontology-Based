# Welcome to the Mentore Help page

**Mentore** is a Graphical User Interface (**GUI**) which lets you easily add subjects, sentences and questions to your ontology.  
Here you will find a quick aid for all the main capabilities of the GUI.  
**Mentore** is supposed to work with the *CKB.owl* ontology, and to operate inside the *schoolSubject* concept: the capabilities of the GUI, and its design, are specifically tailored around them.  
However, the code can be easily adjusted in order let **Mentore** handle different ontologies and concepts, but some features may be unavailable.

---

## General information

All pages feature a link to the this **Help page** on the top-left corner.  
Also, all the pages feature a status bar at the bottom, which shows the currently selected subject.

---

## Pages list

1. **Main page**
2. **Add main page**
3. **Browse page**
4. **Add subject page**
5. **Add sentence page**
6. **Add question page**
7. **Help page** (this one)

### Main page (1/7)

This is the first page you see when first launching Mentore.  
Use this page to access all the functionalities of this tool.  
You will be albe to use the **Recently used subjects** feature once you have selected some of them.

### Add main page (2/7)

This page is a menu that lets you choose which kind of element you wish to add: a new subject, a new sentence, or a new question.  
If the last 2 choices are greyed out, you first have to pick a subject.

### Browse page (3/7)

Inside this page you will be able to examine an updated list of all the subject present in the ontology.  
When you pick one of them, it becomes the currently selected subject.

### Add subject page (4/7)

This page allows you enter a new subject name and to add it to the ontology.  
Once you add the new subject, it gets automatically selected as the currently active one.

### Add sentence page (5/7)

Here you can add a new sentence for the currently active subject.  
In the bottom-left corner of the page, you can choose which kind of sentence to add:

- **Positive sentence**
- **Negative sentence**
- **Wait sentence**

### Add question page (6/7)

Similarly to the previous entry, this page allows you to add a new question (and answer) relative to the currently selected subject.  
You can specify the type of question in the bottom-left portion of the window. The possibilities are:

- **Plain question**
- **Goal question**
- **Contextual question**

Out of the 3 possibilities, **Contextual question** is the only one that requires you to give a paired answer.

### Help page (7/7)

This is where you are now :)

---

## Features that will be supported in the future

The **Mentore GUI** has been developed as part of a broader project, *A Motivational And Entertaining Ontology-based Robotic System For Education*.  
For this reason and due to time constraints, some desiderable features have not been implemented yet:

- The ability of deleting subjects, sentences and questions from the ontology
- A feature that checks whether user-inputted sentences and questions are already present in the ontology in the very same form. This may be tackled with 6 new list attributes that keep track of all the current child data properties of hasSentence for the currently selected subject. This is what is already being done for the list of subjects in *SchoolSubject*
- The possibility of recording some audio files with the GUI, which then may be uploaded online and linked in the ontology

---

## Developers

Andrea Pitto - `s3942710@studenti.unige.it`  
Syed Muhammad Raza Rizvi - `s4853521@studenti.unige.it`  
Laiba Zahid - `s4853477@studenti.unige.it`
