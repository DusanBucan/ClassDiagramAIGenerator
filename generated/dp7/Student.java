package model;

import model.Osoba;

public class Student extends Osoba {

	private String index;
	private double prosek;

	public Student () { }

	public Student (String index, double prosek) {
		this.index = index;
		this.prosek = prosek;
	}


	public void podaci ( ) {
		return null;
	}

}
