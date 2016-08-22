#!/usr/bin/perl

use strict;
use warnings;

open(my $fh, '<:encoding(UTF-8)', 'train.csv')
  or die "Could not open file 'train.csv' $!";

open(TRAIN_DATA, '>train-data.csv');
open(TRAIN_LABELS, '>train-labels.csv');
open(TEST_DATA, '>test-data.csv');
open(TEST_LABELS, '>test-labels.csv');

<$fh>;  # skip the header

my $nTrain = 1;
my $nTest = 1;
my $nTrainWanted = 10000;
my $nTestWanted = 2000;

while (my $row = <$fh>) {
  chomp $row;
  # Notice the "1" in the regex: we are only taking is_booking == 1
  my ($date, $userCity, $distance, $dest, $label) = $row =~ m/^(\d\d\d\d-\d\d).*,(\d+),(\d+\.\d+),(\d+),.*?,1,.*?,.*?,.*?,.*?,(\d+)$/;

  if (defined $distance) {  # actually the fields are either all defined or all undefined after the regex
    if ($nTrain < $nTrainWanted and ($date eq '2014-07' or $date eq '2014-08' or $date eq '2014-09')) {
      print TRAIN_DATA "$userCity,$distance,$dest\n";
      print TRAIN_LABELS "$label\n";
      $nTrain++;
    } elsif ($nTest < $nTestWanted and ($date eq '2014-10' or $date eq '2014-11' or $date eq '2014-12')) {
      print TEST_DATA "$userCity,$distance,$dest\n";
      print TEST_LABELS "$label\n";
      $nTest++;
    }
  }
  last if $nTrain >= $nTrainWanted and $nTest >= $nTestWanted;
}

close TRAIN_DATA;
close TRAIN_LABELS;
close TEST_DATA;
close TEST_LABELS;