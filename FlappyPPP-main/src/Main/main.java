package Main;

import Controleur.Avancer;
import Vue.Affichage;

import javax.swing.*;


/** main
 * La classe main construit un objet de chaque classe (modèle, vue et contrôleur), les relie ensemble, puis elle crée
 * un objet JFrame dans laquelle elle ajoute la vue.
 */


public class main extends JFrame {
	public Avancer avancer;

	public main(String name) {
		Affichage a = new Affichage();
		this.setName(name);
		this.add(a);
		this.pack();
		this.setVisible(true);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		avancer = new Avancer(a);
	}

	public static void main(String[] args) {
		new main("Flappy Bird TEST");
	}
}