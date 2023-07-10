#include "TROOT.h"
#include "TH1.h"
#include "TTree.h"
#include "TH2.h"
#include "TF1.h"
#include "TFile.h"
#include "TMath.h"
#include "TSystem.h"

using namespace std;

using std::cout;
using std::endl;

void root_tutorial_resolution(){

    TFile *file(0);
   TString fname = "./tmva_class_example.root";
   if (!gSystem->AccessPathName( fname )) {
      file = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      TFile::SetCacheFileDir(".");
      file = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!file) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }

    // Let's do some simple tasks with this data:
    // 1- Start of  by plotting the variables in the TTrees (only for signal)
    // 2- Now let's try and make something a bit fancier. Try out the TPads and plot a signal var 
    //    on one pad and the same var on background on another pad
    //    If you dont know how to set a pad look at this example code

    //  TPad *p1 = new TPad("p1","p1",xlow,ylow,xhigh,yhigh);
    //   p1->SetBorderMode(1); 
    //   p1->SetFrameBorderMode(0); 
    //   p1->SetBorderSize(2);
    //   p1->SetBottomMargin(0.10);
    //   p1->Draw();

    //   you can use (0.,0.45,1.,1) and (0.,0.,1.,0.45) as coordinates for the pads
    //  remember to use ->cd() to switch to the pad you wanna draw on!
    // 3- Let's now try and cut our data. Plot var1 on TreeS considering only events with var1>0 or var3<var4  
    // 4- Lastly create a new variable =sqrt(var1**2+var2**2) and write a new root file with two trees, TreeB and 
    //    TreeS with a new branch containing our new variable
    









    file->Close();
}