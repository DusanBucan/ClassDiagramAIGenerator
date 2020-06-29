package model;

import model.Osoba;

public class Student extends Osoba {

	private String ndex;
	private double prosek;

	public Student () { }

	public Student (String ndex, double prosek) {
		this.ndex = ndex;
		this.prosek = prosek;
	}


	public void podaci ( ) {
		return null;
	}

}
