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
/// \file NestedTwistedTrapDetectorConstruction.cc
/// \brief Implementation of the NestedTwistedTrapDetectorConstruction class

#include "NestedTwistedTrapDetectorConstruction.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4TwistedTrap.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4GlobalMagFieldMessenger.hh"
#include "G4AutoDelete.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ThreadLocal 
G4GlobalMagFieldMessenger* NestedTwistedTrapDetectorConstruction::fMagFieldMessenger = nullptr; 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

NestedTwistedTrapDetectorConstruction::NestedTwistedTrapDetectorConstruction(G4double reduction, G4int nNested)
 : B4DetectorConstruction(),
   fReduction(reduction),
   fNNested(nNested),
   fCheckOverlaps(true)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

NestedTwistedTrapDetectorConstruction::~NestedTwistedTrapDetectorConstruction()
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* NestedTwistedTrapDetectorConstruction::Construct()
{
  // Define materials 
  DefineMaterials();
  
  // Define volumes
  return DefineVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void NestedTwistedTrapDetectorConstruction::DefineMaterials()
{ 
  // Lead material defined using NIST Manager
  auto nistManager = G4NistManager::Instance();
  nistManager->FindOrBuildMaterial("G4_Pb");
  
  // Liquid argon material
  G4double a;  // mass of a mole;
  G4double z;  // z=mean number of protons;  
  G4double density; 
  new G4Material("liquidArgon", z=18., a= 39.95*g/mole, density= 1.390*g/cm3);
         // The argon by NIST Manager is a gas with a different density

  // Vacuum
  new G4Material("Galactic", z=1., a=1.01*g/mole,density= universe_mean_density,
                  kStateGas, 2.73*kelvin, 3.e-18*pascal);

  // Print materials
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* NestedTwistedTrapDetectorConstruction::DefineVolumes()
{
  // Geometry parameters
  G4double calorSizeXY  = 4*mm;

  auto worldSizeXY = 1. * calorSizeXY;
  auto worldSizeZ  = 1. * calorSizeXY; 
  
  // Get materials
  auto defaultMaterial = G4Material::GetMaterial("Galactic");
  auto absorberMaterial = G4Material::GetMaterial("G4_Pb");
  auto gapMaterial = G4Material::GetMaterial("liquidArgon");
  
  if ( ! defaultMaterial || ! absorberMaterial || ! gapMaterial ) {
    G4ExceptionDescription msg;
    msg << "Cannot retrieve materials already defined."; 
    G4Exception("NestedTwistedTrapDetectorConstruction::DefineVolumes()",
      "MyCode0001", FatalException, msg);
  }  
   
  //     
  // World
  //
  auto worldS 
    = new G4Box("World",           // its name
                 worldSizeXY/2, worldSizeXY/2, worldSizeZ/2); // its size
                         
  auto worldLV
    = new G4LogicalVolume(
                 worldS,           // its solid
                 defaultMaterial,  // its material
                 "World");         // its name
                                   
  auto worldPV
    = new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(),  // at (0,0,0)
                 worldLV,          // its logical volume                         
                 "World",          // its name
                 0,                // its mother  volume
                 false,            // no boolean operation
                 0);                // copy number
  if (fCheckOverlaps){
		  G4bool overlap = worldPV->CheckOverlaps();
		  if (overlap){
			  G4ExceptionDescription msg;
			  msg << "You have overlapping volumes!" << G4endl;
			  msg << " World volume properties are: "  << worldSizeXY/2 << " " << worldSizeZ/2 << G4endl;
			  G4Exception("NestedTwistedTrapDetectorConstruction::DefineVolumes()",
			              "MyCode0002", FatalException, msg);
		  }
	  }
  
  //                               
  // Calorimeter
  //
  
  G4double pDx1 = (calorSizeXY*0.6)/2;
  G4double pDx2 = (calorSizeXY*0.3)/4;
  G4double pDy = calorSizeXY/4;
  G4double pDz = calorSizeXY/2;
  G4double twistAng = 60;
  auto calorimeterS
	  = new G4TwistedTrap("CalorimeterTwistedTrap",
                          twistAng*deg,
                          pDx1,  // half x length at -pDz,-pDy
                          pDx2,  // half x length at -pDz,+pDy
                          pDy,  // half y
                          pDz); // half z
  std::cout << "The largest scales are: " <<  pDx1 << ", " << pDx2 << ", " << pDy << ", " << pDz << std::endl;


  auto calorLV
    = new G4LogicalVolume(
                 calorimeterS,     // its solid
                 defaultMaterial,  // its material
                 "Calorimeter");   // its name
                                   
  auto placement = new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(),  // at (0,0,0)
                 calorLV,          // its logical volume                         
                 "Calorimeter",    // its name
                 worldLV,          // its mother  volume
                 false,            // no boolean operation
                 0);                // copy number

  if (fCheckOverlaps){
		  G4bool overlap = placement->CheckOverlaps();
		  if (overlap){
			  G4ExceptionDescription msg;
			  msg << "You have overlapping volumes!" << G4endl;
			  msg << " Calorimeter twisted trap volume properties are: " << twistAng << " " << pDx1 << " " << pDx2 << " " << pDy << " " << pDz << G4endl;
			  G4Exception("NestedTwistedTrapDetectorConstruction::DefineVolumes()",
			              "MyCode0002", FatalException, msg);
		  }
	  }
  G4ThreeVector pMin, pMax;
  calorLV->GetSolid()->BoundingLimits(pMin, pMax);
  fXMin = pMin.x();
  fYMin = pMin.y();
  fZMin = pMin.z();

  fXMax = pMax.x();
  fYMax = pMax.y();
  fZMax = pMax.z();
  
  //twistAng = 30;
  auto motherVol = calorLV;
  for (int nestI = 0; nestI < fNNested; nestI++ ){
	  std::string trapName = "InnerCalorimeterTwistedTrap"+std::to_string(nestI);
	  std::string logicalTrapName = "InnerCalorimeter"+std::to_string(nestI);

	  pDx1 *= fReduction;
	  pDx2 *= fReduction;
	  pDy *= fReduction;
	  //pDz *= fReduction;

	  auto InnerCalorimeterS
		  = new G4TwistedTrap(trapName,
		                      twistAng*deg,
		                      pDx1,  // half x length at -pDz,-pDy
		                      pDx2,  // half x length at -pDz,+pDy
		                      pDy,  // half y
		                      pDz); // half z

	  auto InnerCalorLV
		  = new G4LogicalVolume(
		                        InnerCalorimeterS,     // its solid
		                        defaultMaterial,       // its material
		                        logicalTrapName);   // its name
                                   
	  placement = new G4PVPlacement(
	                    0,                   // no rotation
	                    G4ThreeVector(),     // at (0,0,0)
	                    InnerCalorLV,        // its logical volume                         
	                    logicalTrapName,  // its name
	                    motherVol,             // its mother  volume
	                    false,               // no boolean operation
	                    0); // copy number
	  if (fCheckOverlaps){
		  G4bool overlap = placement->CheckOverlaps();
		  if (overlap){
			  G4ExceptionDescription msg;
			  msg << "You have overlapping volumes!" << G4endl;
			  msg << " Volume properties are: " << trapName << " " << twistAng << " " << pDx1 << " " << pDx2 << " " << pDy << " " << pDz << G4endl;
			  G4Exception("NestedTwistedTrapDetectorConstruction::DefineVolumes()",
			              "MyCode0002", FatalException, msg);
		  }
	  }
	  motherVol = InnerCalorLV;
  }
  std::cout << "The smallest scales are: " <<  pDx1 << ", " << pDx2 << ", " << pDy << ", " << pDz << std::endl;
  //                                        
  // Visualization attributes
  //
  worldLV->SetVisAttributes (G4VisAttributes::GetInvisible());

  auto simpleBoxVisAtt= new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  simpleBoxVisAtt->SetVisibility(true);
  calorLV->SetVisAttributes(simpleBoxVisAtt);

  //
  // Always return the physical World
  //
  return worldPV;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void NestedTwistedTrapDetectorConstruction::ConstructSDandField()
{ 
  // Create global magnetic field messenger.
  // Uniform magnetic field is then created automatically if
  // the field value is not zero.
  G4ThreeVector fieldValue;
  fMagFieldMessenger = new G4GlobalMagFieldMessenger(fieldValue);
  fMagFieldMessenger->SetVerboseLevel(1);
  
  // Register the field messenger for deleting
  G4AutoDelete::Register(fMagFieldMessenger);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
