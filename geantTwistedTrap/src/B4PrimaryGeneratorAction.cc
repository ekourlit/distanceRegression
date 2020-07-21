//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// 
/// \file B4PrimaryGeneratorAction.cc
/// \brief Implementation of the B4PrimaryGeneratorAction class

#include "B4PrimaryGeneratorAction.hh"

#include "G4RunManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "time.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4PrimaryGeneratorAction::B4PrimaryGeneratorAction()
	: G4VUserPrimaryGeneratorAction(),
	  fParticleGun(nullptr)
{
  
	G4int nofParticles = 1;
	fParticleGun = new G4ParticleGun(nofParticles);

	// default particle kinematic
	//
	auto particleDefinition = G4ParticleTable::GetParticleTable()->FindParticle("geantino");
	fParticleGun->SetParticleDefinition(particleDefinition);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4PrimaryGeneratorAction::~B4PrimaryGeneratorAction()
{
	delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
	float fXMin = 0;
	float fXMax = 10000; // CHANGE THIS
	float fYMin = 0;
	float fYMax = 10000; // CHANGE THIS
	float fZMin = 0;
	float fZMax = 10000; // CHANGE THIS
	
	float xLength = fabs(fXMax-fXMin);
	float yLength = fabs(fYMax-fYMin);
	float zLength = fabs(fZMax-fZMin);
  
	float x, y, z, dx, dy, dz;
	x = fXMin+G4UniformRand()*xLength;
	y = fYMin+G4UniformRand()*yLength;
	z = fZMin+G4UniformRand()*zLength;

	dx = G4UniformRand();
	dy = G4UniformRand();
	dz = G4UniformRand();
  
	fParticleGun->SetParticlePosition(position);
	fParticleGun->SetParticleMomentumDirection(G4ThreeVector(dx,dy,dz));
	fParticleGun->GeneratePrimaryVertex(anEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

