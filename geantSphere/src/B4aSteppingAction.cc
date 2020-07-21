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
/// \file B4aSteppingAction.cc
/// \brief Implementation of the B4aSteppingAction class

#include "B4aSteppingAction.hh"
#include "B4aEventAction.hh"
#include "B4DetectorConstruction.hh"
#include "B4Analysis.hh"

#include "G4Step.hh"
#include "G4RunManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aSteppingAction::B4aSteppingAction(
                                     const B4DetectorConstruction* detectorConstruction,
                                     B4aEventAction* eventAction,
                                     B4RunAction* runAction)
	: G4UserSteppingAction(),
	  fDetConstruction(detectorConstruction),
	  fEventAction(eventAction),
	  fRunAction(runAction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aSteppingAction::~B4aSteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4aSteppingAction::UserSteppingAction(const G4Step* step)
{
	// Collect energy and track length step by step

	// get analysis manager
	auto analysisManager = G4AnalysisManager::Instance();

	// step length
	G4double stepLength = step->GetStepLength();;
	auto prepoint = step->GetPreStepPoint();
	auto momentumDirection = prepoint->GetMomentumDirection();

	G4double startX = prepoint->GetPosition().x();
	G4double startY = prepoint->GetPosition().y();
	G4double startZ = prepoint->GetPosition().z();
  
	G4double dirX = momentumDirection.x();
	G4double dirY = momentumDirection.y();
	G4double dirZ = momentumDirection.z();
  
	auto lengthHist = analysisManager->GetH1(0);
	// Check that all bins have the minimum yield;
	G4float minYield = 9999;
	G4int lengthBin = -99;
	G4int binYield = 0;
	G4float lEdge = 0;
	G4float uEdge = 0;
	// including both under and overlfow
	uint nBins = lengthHist->axis().bins();
	for (uint binI=0; binI<nBins; binI++){
		binYield = lengthHist->bin_entries(binI);
		
		// Find minimum bin yield
		if (binYield < minYield)
			minYield = binYield;

		// Find histogram bin of current length.
		lEdge = lengthHist->axis().bin_lower_edge(binI);
		uEdge = lengthHist->axis().bin_upper_edge(binI);
		if (stepLength < lengthHist->axis().bin_lower_edge(0))
			lengthBin = 0;
		else if (stepLength > lengthHist->axis().bin_upper_edge(nBins-1))
			lengthBin = nBins;
		else if (stepLength > lEdge && stepLength <= uEdge)
			lengthBin = binI;
	}

	// This should only matter at high stats. 
	G4float minYieldWithBuff = 0;
	if (minYield > 0)
		minYieldWithBuff = minYield*fRunAction->fAllowedDiff;
	G4int lengthBinEntries = lengthHist->bin_entries(lengthBin);

	if (lengthBinEntries <= minYieldWithBuff){
		// fill ntuple
		analysisManager->FillNtupleDColumn(0, startX);
		analysisManager->FillNtupleDColumn(1, startY);
		analysisManager->FillNtupleDColumn(2, startZ);
		analysisManager->FillNtupleDColumn(3, dirX);
		analysisManager->FillNtupleDColumn(4, dirY);
		analysisManager->FillNtupleDColumn(5, dirZ);
		analysisManager->FillNtupleDColumn(6, stepLength);
		analysisManager->AddNtupleRow();

		analysisManager->FillH1(0, stepLength);
	}
	// do only one step
	G4Track *track = step->GetTrack();
	int stepNum = track->GetCurrentStepNumber();
	if (stepNum > 0)
		track->SetTrackStatus(fStopAndKill);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
