# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:33:42 2020
""" 
print('début')

#Artificial intelligence for business
#Optimizing Warehouse Flows with Q-Learning

# Importing the libraries
import numpy as np

# paramètres
gamma = 0.75
alpha = 0.9

# Structure du code (plan/stratégie)

# PART 1 - définition de l'environnement

#Etats

# Création d'un dico
location_to_state = {"A":0, 
                     "B":1,
                     "C":2,
                     "D":3,
                     "E":4,
                     "F":5,
                     "G":6,
                     "H":7,
                     "I":8,
                     "J":9,
                     "K":10,
                     "L":11}

state_to_location = {state: location for location, state in location_to_state.items()}

#Actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]


# PART 3 - Goint into production (automatisation de la partie 2)

def route(starting_location, ending_location):

    #on stocke le chemin dans une liste initialisée à starting_location
    route = [starting_location]
    
    #Récompenses
    R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                  [1,0,1,0,0,1,0,0,0,0,0,0],
                  [0,1,0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0],
                  [0,1,0,0,0,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,1,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1,0,1],
                  [0,0,0,0,0,0,0,1,0,0,1,0]])    

    #on va modifier R pour que R[ending_location, ending_location] = 1000
    ending_state = location_to_state[ending_location]
    R[ending_state,ending_state]=1000   #on met 1000 pour indiquer l'objectif
         
    #L'apprentissage est placé ici
    # PART 2 - Construction de l'ia - ai solution with q-learning (approche itérative) 

    #   Initialisation des valeurs Q
    Q = np.zeros([12,12])

    #apprentissage
    for _ in range(1000):
        
        current_state = np.random.randint(0, 12)
        #print("current_state=", current_state)
        playable_action = []
        
        #les 12 actions possibles
        for j in range(12):
            if R[current_state,j]>0:
                playable_action.append(j)
        
                #prochain état : un choix aléatoire parmi toutes les actions possibles
                next_state = np.random.choice(playable_action)
                #print("actions possibles pour j=",j,"=>", playable_action)
        #calcul de la dt
        TD =   R[current_state, next_state] \
             + gamma * Q[next_state, np.argmax(Q[next_state,])] \
             - Q[current_state, next_state]

        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    #boucle while
    next_location = starting_location #initialisation
    while next_location != ending_location:
        #on prend une action depuis là où on est
        #on utilise la matrice Q
        #il faut la localisation initiale (lettres)
        #et l'état initial (chiffres)
        starting_state = location_to_state[starting_location] #traduction lettre en chiffre
        #on va prendre une décision sur la ligne correspondante
        #on prend l'indice qui donne la valeur max
        next_state = np.argmax(Q[starting_state,])
        #on traduit nombre en lettre
        #pour cela on a créé le dico inverse state_to_location
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location #on est là maintenant
    
    print('fin route()')

    print(R[ending_state,ending_state])
    
    return route    


#Création d'une fonction de double appel de route
def route2(starting_location, ending_location, intermediate_location):
    #1-appel avec start et interm
    routeA = route(starting_location, intermediate_location)
    print("routeA=",routeA)
    #  sauvegarde du résultat 
    #2-appel avec interm et ending
    routeB = route(intermediate_location, ending_location)
    print("routeB=", routeB)
    #  append du premier résultat avec le deuxième = résultat final
    # on enlève la dernière étape de routeA pour éviter la redondance
    routeA.pop(len(routeA)-1)
    route_finale = routeA +routeB
    print("route_finale=", route_finale)
    return route_finale
    
#LANCEMENT DE LA FONCTION

print("fonction started")
print(route2("A","G","E"))
print("fonction ended")

# fin
"""
for i in range(12):
    print("")
    mystr=""
    for j in range(12):
        mystr = mystr + str(Q[i,j]) + "|"
    print(mystr)
"""

print('fin')