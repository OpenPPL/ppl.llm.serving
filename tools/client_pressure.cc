#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "llm.grpc.pb.h"
#include <grpc++/grpc++.h>
#include <chrono>
const std::vector<std::string> prompts = {
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    // "Here's a possible response:\n\nR\u00e9ponse:\n\nLors de l'utilisation de SQL, il est important de comprendre les basiques pour pouvoir tirer le meilleur parti de ce langage de programmation de base de donn\u00e9es. Voici une fiche rappel organis\u00e9e de mani\u00e8re claire pour vous aider.\n\n1. Syntaxe de base:\nSQL est compos\u00e9 de commandes qui sont utilis\u00e9es pour acc\u00e9der et manipuler des donn\u00e9es dans une base de donn\u00e9es. Les commandes les plus courantes sont SELECT, INSERT, UPDATE et DELETE.\n2. Les requ\u00eates SELECT:\nLa requ\u00eate SELECT est l'une des commandes les plus courantes en SQL. Elle est utilis\u00e9e pour r\u00e9cup\u00e9rer des donn\u00e9es dans une base de donn\u00e9es. Pour utiliser SELECT, il faut sp\u00e9cifier les colonnes que vous souhaitez r\u00e9cup\u00e9rer et la table \u00e0 partir de laquelle vous voulez r\u00e9cup\u00e9rer les donn\u00e9es.\n3. Les filtres WHERE:\nLa clause WHERE est utilis\u00e9e pour filtrer les donn\u00e9es dans une requ\u00eate SELECT. Elle permet de s\u00e9lectionner des enregistrements qui r\u00e9pondent \u00e0 un ou plusieurs crit\u00e8res sp\u00e9cifiques. Les crit\u00e8res peuvent inclure des conditions logiques comme AND et OR.\n4. Les jointures de table:\nLes jointures de table sont utilis\u00e9es pour combiner les donn\u00e9es de deux ou plusieurs tables en une seule requ\u00eate. Les jointures les plus courantes sont INNER JOIN, LEFT JOIN, RIGHT JOIN et FULL OUTER JOIN.\n5. Les agr\u00e9gats de donn\u00e9es:\nLes agr\u00e9gats de donn\u00e9es sont utilis\u00e9s pour calculer des valeurs \u00e0 partir d'un ensemble de donn\u00e9es. Les fonctions les plus courantes sont SUM, COUNT, AVG, MIN et MAX.\n6. La cr\u00e9ation et la gestion de tables:\nSQL peut \u00e9galement \u00eatre utilis\u00e9 pour cr\u00e9er et g\u00e9rer des tables dans une base de donn\u00e9es. Les commandes les plus courantes sont CREATE TABLE, ALTER TABLE et DROP TABLE.\n\nPour une organisation encore meilleure, n'h\u00e9sitez pas \u00e0 utiliser des zones de code pour mettre en \u00e9vidence les exemples de syntaxe SQL. De plus, pour une meilleure compr\u00e9hension, n'h\u00e9sitez pas \u00e0 inclure des exemples concrets d'utilisation de SQL. Bonne chance pour votre interview !\n\n[[1](https://www.indeed.com/career-advice/career-development/memo-examples)]\n[[2](https://www.grammarly.com/blog/how-to-write-memo/)]\n[[3](https://www.indeed.com/career-advice/career-development/memo)]"
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    "Building a website can be done in 10 simple steps:\n",
    "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    "Building a website can be done in 10 simple steps:\n",
    "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    // "I believe the meaning of life is",
    // "Simply put, the theory of relativity states that ",
    // "Building a website can be done in 10 simple steps:\n",
    // "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
};





//  avg genlen = 1152
const std::vector<int> generation_len = {
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
};


using namespace grpc;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;
using namespace ppl::llm;

ABSL_FLAG(std::string, target, "localhost:50052", "Server address");

class GenerationClient {
public:
    GenerationClient(std::shared_ptr<Channel> channel) : stub_(proto::LLMService::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    int Generation(int start_tid = 0) {
        // Data we are sending to the server.
        std::unordered_map<int, std::string> rsp_stream_store;
        std::unordered_map<int64_t, proto::Request*> tid_map;

        std::vector<proto::Request*> req_queue(prompts.size());

        for (size_t i = 0; i < prompts.size(); ++i) {
            proto::Request* req = new proto::Request();
            req->set_id(start_tid + i);
            req->set_prompt(prompts[i]);
            auto* choosing_parameter = req->mutable_choosing_parameters();
            choosing_parameter->set_do_sample(false);
            choosing_parameter->set_temperature(1.f);
            choosing_parameter->set_repetition_penalty(1.f);
            choosing_parameter->set_presence_penalty(0.f);
            choosing_parameter->set_frequency_penalty(0.f);

            auto* stopping_parameters = req->mutable_stopping_parameters();
            stopping_parameters->set_max_new_tokens(512);
            stopping_parameters->set_ignore_eos_token(false);
            tid_map.emplace(req->id(), req);

            req_queue[i] = req;
        }

        int finished_cnt = 0;

        while (!req_queue.empty()) {
            std::cout << "req_queue.size(): " << req_queue.size() << std::endl;
            proto::BatchedRequest req_list;
            for (size_t i = 0; i < req_queue.size(); i++) {
                auto req = req_list.add_req();
                req->set_id(req_queue[i]->id());
                req->set_prompt(req_queue[i]->prompt());
                auto* choosing_parameter = req->mutable_choosing_parameters();
                choosing_parameter->set_do_sample(false);
                choosing_parameter->set_temperature(1.f);
                choosing_parameter->set_repetition_penalty(1.f);
                choosing_parameter->set_presence_penalty(0.f);
                choosing_parameter->set_frequency_penalty(0.f);

                auto* stopping_parameters = req->mutable_stopping_parameters();
                stopping_parameters->set_max_new_tokens(req_queue[i]->stopping_parameters().max_new_tokens());
                stopping_parameters->set_ignore_eos_token(false);
            }
            req_queue.clear();

            ClientContext context;
            std::unique_ptr<ClientReader<proto::BatchedResponse> > reader(stub_->Generation(&context, req_list));

            proto::BatchedResponse batched_rsp;
            while (reader->Read(&batched_rsp)) {
                for (const auto& rsp : batched_rsp.rsp()) {
                    int tid = rsp.id();
                    if (rsp.status() == proto::FAILED) {
                        std::cout << "failed tid: " << tid << std::endl;
                        req_queue.push_back(tid_map.find(tid)->second);
                        continue;
                    }
                    if (rsp.status() == proto::FINISHED) {
                        finished_cnt++;
                    }
                    std::string rsp_stream = rsp.generated();
                    // std::cout << rsp_stream  << std::endl;
                    rsp_stream_store[tid] += rsp_stream;
                }
            }
            Status status = reader->Finish();
            if (status.ok()) {
                std::cout << "Generation rpc succeeded." << std::endl;
            } else {
                std::cerr << "Generation rpc failed." << std::endl;
                return -1;
            }
        }

        std::cout << "Answer: -----------------" << std::endl;
        for (auto& rsp : rsp_stream_store) {
            std::cout << "tid " << rsp.first << ": " << rsp.second << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        std::cout << "finished cnt: " << finished_cnt << std::endl;
        return 0;
    }

private:
    std::unique_ptr<proto::LLMService::Stub> stub_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " host:port" << std::endl;
        return -1;
    }

    const std::string target_str = argv[1];
    int start_tid = 0;
    if (argc == 3)
        start_tid = std::stoi(std::string(argv[2]));

    GenerationClient generator(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

    generator.Generation(start_tid);
    return 0;
}