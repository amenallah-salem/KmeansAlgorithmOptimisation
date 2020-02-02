
# -*- coding: utf-8 -*-


from pyspark import SparkContext, SparkConf
from math import sqrt
import time

def computeDistance(x,y):
    return sqrt(sum([(a - b)**2 for a,b in zip(x,y)]))


def closestCluster(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster,min_dist)

def sumList(x,y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenneList(x,n):
    return [x[i]/n for i in range(len(x))]

def Kmeanspp(data, nb_clusters):
    clusteringDone = False
    number_of_steps = 0
    current_error = float("inf")
    # A broadcast value is sent to and saved  by each executor for further use
    # instead of being sent to each executor when needed.
    nb_elem = sc.broadcast(data.count())

    #############################
    # Select initial centroides #
    #############################
    #centroides = sc.parallelize(data.takeSample('withoutReplacment',nb_clusters)).zipWithIndex().map(lambda x: (x[1],x[0][1][:-1]))
    # (0, [4.4, 3.0, 1.3, 0.2])
    # In the same manner, zipWithIndex gives an id to each cluster

    centroides = sc.parallelize(data.takeSample('withoutReplacment',1))\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0][1][:-1]))
    print("Nombre de partitions des centroides : %s" % centroides.getNumPartitions())
    #changer nb_cluster par 1 : voir rappor :
    #\iti Choisissez la première moyenne $\mu_1$ \textbf{au hasard}
    #dans l'ensemble $X=$ $\{x_1,\ldots, x_k\}$ et
    #ajoutez-la à l'ensemble $M=\{\mu_1,\ldots, \mu_k\}$.

    ###################################
    ###################################
    #\itii Pour chaque point $x \in X$,
    #calculez la distance au carré $D (x)$
    #entre $x$ et la moyenne la plus proche en $M$.
    ###################################



    while not clusteringDone:

        #############################
        # Assign points to clusters #
        #############################
        #pour mieu accelerer le calcule on enlève la colone
        #'Iris-setosa' avant quand fait le produit cartésien
        #Par cette méthode en enlève une colonne de la boucle



        print("Le clustering commence: ")
        print('Itération numero : '+str(number_of_steps))
        joined = data.cartesian(centroides)
        print("Nombre de partitions de joined : %s" % joined.getNumPartitions())
        joined = joined.coalesce(3)
        print("Nombre de partitions de joined après coalesce : %s" % joined.getNumPartitions())

        # We compute the distance between the points and each cluster
        dist = joined.map(lambda x: (x[0][0],(x[1][0], computeDistance(x[0][1][:-1], x[1][1]))))
        # (0, (0, 0.866025403784438))
        print("Nombre de partitions de dist : %s" % dist.getNumPartitions())

        dist_list = dist.groupByKey().mapValues(list)
        print("Nombre de partitions de dist_list : %s" % dist_list.getNumPartitions())

        # We keep only the closest cluster to each point.
        min_dist = dist_list.mapValues(closestCluster)
        print("Nombre de partitions de min_dist : %s" % min_dist.getNumPartitions())

        # assignment will be our return value : It contains the datapoint,
        # the id of the closest cluster and the distance of the point to the centroid
        assignment = min_dist.join(data)
        #Remarque: Dans cette étape on vas récuperé la colonne de nom du fleure : ex:Iris-setosa
        # (0, ((2, 0.5385164807134504), [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']))
        print("Nombre de partitions de assignment : %s" % assignment.getNumPartitions())

        ############################################
        # Compute the new centroid of each cluster #
        ############################################

        clusters = assignment.map(lambda x: (x[1][0][0], x[1][1][:-1]))
        # (2, [5.1, 3.5, 1.4, 0.2])
        #Remarque : count  compte le nombre de vecteurs par centroide
        count = clusters.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y)

        #Maintenain : on somme les vecteurs dans chaque centroide
        somme = clusters.reduceByKey(sumList)
        centroidesCluster = somme.join(count).map(lambda x : (x[0],moyenneList(x[1][0],x[1][1])))
        print("Nombre de partitions de centroidesCluster : %s" % centroidesCluster.getNumPartitions())

        ############################
        # Is the clustering over ? #
        ############################

        # Let's see how many points have switched clusters.
        if number_of_steps > 0:
            switch = prev_assignment.join(min_dist)\
                                    .filter(lambda x: x[1][0][0] != x[1][1][0])\
                                    .count()
        else:
            switch = 150
        if switch == 0 or number_of_steps == 100:
            clusteringDone = True
            error = sqrt(min_dist.map(lambda x: x[1][1]).reduce(lambda x,y: x + y))/nb_elem.value
        else:
            centroides = centroidesCluster
            prev_assignment = min_dist
            number_of_steps += 1
            print("fin boucle: ")
            print("Temps d execution : %s secondes ---" % (time.time() - startTime))

    return (assignment, error, number_of_steps)


if __name__ == "__main__":

    conf = SparkConf().setAppName('Kmeanspp')
    sc = SparkContext(conf=conf)

    lines = sc.textFile("hdfs:/user/user81/iris.data.txt")
    data = lines.map(lambda x: x.split(','))\
            .map(lambda x: [float(i) for i in x[:4]]+[x[4]])\
            .zipWithIndex()\
            .map(lambda x: (x[1],x[0]))

    startTime = time.time()
    print('En a commencer à' , startTime)
    clustering = Kmeanspp(data,3)

    #clustering[0].saveAsTextFile("hdfs:/user/user81/output")
    clustering[0].coalesce(1).saveAsTextFile("hdfs:/user/user81/output")
    print("Le nbr d etape est de :" ,clustering[2])

    print(clustering)
    #On mesure le temps d'exécution :
    print("Temps d execution de l'algorithme : %s secondes ---" % (time.time() - startTime))