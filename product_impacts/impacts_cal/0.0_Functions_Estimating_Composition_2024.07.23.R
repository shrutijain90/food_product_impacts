#######
# Original code from: Clark et al., 2022, Estimating the environmental impacts of 57,000 food products
#######


#######
# This file contains functions to pair a database containing back-of-package ingredient lists
# To environmental databases
# And health databases
# Such that the output is an estimate of the env or health impacts per food product
#######


###
# Function to find and extract embedded ingredients in a data frame
embedded.function = 
  function(dat,
           num_embedded,
           ingredient_list,
           rows.loop, 
           embedded) {
    ###
    # Creating column that contains ingredient text
    dat$ingredients_text_col =
      dat[,ingredient_list]
    
    ###
    # Finding start and stop location of embedded ingredients in a recipe
    # Doing this in a loop
    # Because this was easier to make work
    
    # Creating empty lists to append the strings to
    # But only doing this if keeping the extracted text
    if(embedded != 'embedded') {
      for(i in 1:num_embedded) {
        dat[,paste0('string',i)] = NA
      }
      
      for(i in 1:num_embedded) {
        dat[,paste0('ing_string',i)] = NA
      }
      
    }
    
    # And looping over each row
    for(i in rows.loop) {
      # for(i in 1:10) {
      
      for(j in 1:num_embedded) {
        num.char = str_length(dat$ingredients_text_col[i])
        start.1 = c()
        end.1 = c()
        start.2 = c()
        end.2 = c()
        
        tot.open = str_count(dat$ingredients_text_col[i],"\\(")
        tot.close = str_count(dat$ingredients_text_col[i],"\\(")
        
        #####
        ###
        # Method one:
        # Identify first close parentheses that is not preceded immediately by a % sign
        # Finding end location
        end.1 = 
          min(c(str_locate(dat$ingredients_text_col[i],
                           "[A-z](\\s)?\\)(\\s?),(\\s)?[A-z]|\\)(\\s)?\\)(\\s)?,(\\s)?[A-z]")+1,
                str_locate(dat$ingredients_text_col[i],
                           "\\([A-z]*\\)")[2],
                str_locate(dat$ingredients_text_col[i],
                           "[A-z](\\s)\\)(\\s)?,(\\s)?\\*(\\s)?[A-z]")[1]+2,
                str_locate(dat$ingredients_text_col[i],
                           "[A-z]\\)(\\s)?,(\\s)?\\*(\\s)?[A-z]")[1]+1), na.rm= TRUE)
        
        # Getting location of open parentheses before the close parentheses identified above
        start.1 = str_locate_all(substr(dat$ingredients_text_col[i],1,end.1),"\\(") %>%
          unlist(.) %>% .[1:(length(.)/2)]
        
        k = length(start.1)
        open.1 = str_count(substr(dat$ingredients_text_col[i], start.1[k], end.1), "\\(")
        close.1 = str_count(substr(dat$ingredients_text_col[i], start.1[k], end.1), "\\)")
        
        # Logical checks to see if number of open and close parentheses before start.1 are not equal to each other
        # If they are equal to each other, then there is probably a typo in the ingredient list
        # In addition, also checking to make sure sum of open and close parentheses is not equal
        logic.check = 
          str_count(substr(dat$ingredients_text_col[i], 1, start.1[k]-1), "\\(") == str_count(substr(dat$ingredients_text_col[i], 1, start.1[k]-1), "\\)") &
          str_count(dat$ingredients_text_col[i], "\\(") != str_count(dat$ingredients_text_col[i], "\\)")
        
        # Looping back through earlier section of ingredient list
        # To identify earlier open parentheses
        # Then rechecking if # open == # close
        while(k > 0 & open.1 != close.1 &
              !is.na(open.1) & !is.na(close.1) &
              !logic.check) {
          open.1 = str_count(substr(dat$ingredients_text_col[i], start.1[k], end.1), "\\(")
          close.1 = str_count(substr(dat$ingredients_text_col[i], start.1[k], end.1), "\\)")
          
          if(open.1 == close.1) {
            k = k
          } else if(open.1 != close.1) {
            k = k-1
          }
        }
        
        # K cannot = 0
        if(k %in% 0) {
          k = 1
        }
        
        # Saving for later use
        start.1 = start.1[k]
        # Quick check: Sometimes returns "intger(0)"
        # This indicates it did not find the string
        # Converting this value to -1
        if(length(start.1) == 0 | is.na(start.1)) {
          start.1 = -1
        }
        
        #####
        ###
        # Method 2: 
        # Identify open parentheses first, and then identify close parentheses after
        start1 = 
          str_locate(dat$ingredients_text_col[i],
                     "\\([0-9]{1,2}%\\)(\\s)?\\([^0-9]")[[2]]-1
        
        start2 = 
          str_locate(dat$ingredients_text_col[i], 
                     "[A-z](\\s)?\\([^0-9]")[[2]]-1
        
        start3 =
          str_locate(dat$ingredients_text_col[i],
                     "\\([0-9]{1,2}(\\.)[0-9]{1,2}%\\)(\\s)?\\([^0-9]")[[2]]-1
        
        start4 =
          str_locate(dat$ingredients_text_col[i],
                     "[0-9]{1,2}%(\\s)?\\([^0-9]")[[2]]-1
        
        start5 =
          str_locate(dat$ingredients_text_col[i],
                     "[0-9]{1,2}(\\.)[0-9]{1,2}%(\\s)?\\([^0-9]")[[2]]-1
        
        start6 = 
          str_locate(dat$ingredients_text_col[i],
                     "\\([0-9]{1,2}%\\)(\\,)?(\\s)?\\([^0-9]")[[2]]-1
        
        start7 =
          str_locate(dat$ingredients_text_col[i],
                     "in varying proportions\\) \\(Courgette")[[2]] - 9
        
        start8 = 
          str_locate(dat$ingredients_text_col[i],
                     "(\\s)[0-9]{1,2}(\\s)%(\\s)?\\([^0-9]")[[2]]-1
        
        start9 = 
          str_locate(dat$ingredients_text_col[i],
                     "(\\s)[0-9]{1,2},[0-9]{1,2}(\\s)%(\\s)?\\([^0-9]")[[2]]-1
        
        start10 = 
          str_locate(dat$ingredients_text_col[i],
                     "(\\s)[0-9]{1,2}(\\s)%(\\s)?\\([^0-9]")[[2]]-1
        
        start11 = 
          str_locate(dat$ingredients_text_col[i],
                     "(\\s)[0-9]{1,2}(\\.)[0-9]{1,2}(\\s)%(\\s)?\\([^0-9]")[[2]]-1
        
        start12 =
          str_locate(dat$ingredients_text_col[i],
                     "(\\s)\\([0-9]{1,2}%,(\\s)[A-Z]")[[1]]+1
        
        start13 =
          str_locate(dat$ingredients_text_col[i],
                     "\\([0-9]{1,2}(\\s)%\\)(\\s)\\([A-z]")[[2]]-2
        
        start14 = 
          str_locate(dat$ingredients_text_col[i],
                     "Vegetables(\\s)\\([0-9].[0-9]%")[[2]]-4
        
        start15 = 
          str_locate(dat$ingredients_text_col[i],
                     "\\([0-9]{1,2}%\\):(\\s)\\([A-z]")[[2]]-2
        
        start16 = 
          str_locate(dat$ingredients_text_col[i],
                     "Vegetables(\\s)\\([0-9][0-9].[0-9]%")[[2]]-5
        
        start17 = 
          str_locate(dat$ingredients_text_col[i],
                     "over dr. weight\\) \\([A-z]")[[2]]-1
        
        start18 = 
          str_locate(dat$ingredients_text_col[i],
                     "Brown Flour \\(50%")[[2]]-3
        
        start19 = 
          str_locate(dat$ingredients_text_col[i],
                     "Gherkins, \\(Gh")[[2]]-3
        
        start20 =
          str_locate(dat$ingredients_text_col[i],
                     "Chocolate \\([0-9]{1,2}%\\)(\\s)-(\\s)\\(Sug")[[2]]-3
        
        start21 =
          str_locate(dat$ingredients_text_col[i],
                     "%,(\\s)\\([A-z]")[[2]]-2
        
        start22 =
          str_locate(dat$ingredients_text_col[i],
                     "\\(Edible Wafer")[[1]]
        
        start23 = 
          str_locate(dat$ingredients_text_col[i],
                     "\\(100%\\)(\\s)\\([A-z]")[2]-1

        
        
        # Taking min or these, excluding NAs
        start.2 = min(c(start1, start2, start3, start4, start5, 
                        start6, start7, start8, start9, start10,
                        start11, start12, start13, start14, start15,
                        start16, start17, start18, start19, start20,
                        start21, start22, start23), 
                      na.rm = TRUE)
        
        ###
        # Only doing this if start.2 != start.1
        # If they're equal, then then they'll return the same value
        if(start.1 != start.2 & 
           !is.na(start.1) & !is.na(start.2) &
           is.finite(start.1) & is.finite(start.2)) {
          
          # if(!is.na(start12) &
          #    start12 == start.2) {
          #   start.12.list = c(start.12.list, i)
          # }
          # 
          # Finding end location
          end.2 = 
            min(str_locate(substr(dat$ingredients_text_col[i],start.2,num.char),
                           "\\)(\\s)?,(\\s)?[A-z]")[[1]] + 1 + start.2, na.rm = TRUE)
          # Checking if this returns an infinite value
          if(!is.finite(end.2)) {
            end.2 = start.2 + 1
          }
          
          # Getting number of open and close parentheses
          # These should be the same
          open.par = 
            str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                      "\\(")
          close.par = 
            str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                      "\\)")
          
          # Logical check, close parentheses cannot be larger than open parentheses
          # If it is, identify location of first previous open parentheses
          # Doing this in a while loop
          # But adding a counter to avoid an indefinite loop
          counter = 1
          while(open.par < close.par & counter <= 25 & !is.na(open.par) & !is.na(close.par)) {
            # Adding one to counter
            counter = counter + 1
            # Identifying first previous open parentheses
            tmp.start = str_locate_all(substr(dat$ingredients_text_col[i],1,start.2), "\\(")
            tmp.start =  max(tmp.start[[1]][tmp.start[[1]][,1] < start.2])
            start = tmp.start
            
            open.par = 
              str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                        "\\(")
            close.par = 
              str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                        "\\)")
          }

          # If not, then search for next close parentheses
          # count number of open and close parentheses
          # Then repeat until this is true
          tmp.end.2 = 0
          if(!is.na(end.2) & is.finite(end.2)) {
            counter = 1
            while(close.par != open.par & counter < 25 & is.finite(tmp.end.2)) {
              counter = counter + 1
              tmp.end.2 = min(str_locate(substr(dat$ingredients_text_col[i], end.2, num.char),
                                         "\\)")[[2]], na.rm = TRUE)
              if(is.infinite(tmp.end.2)) {
                end.2 = num.char
              } else {
                # Exception if last characters are two parentheses, but only if total number of parentheses in the character string is uneven
                if(substr(dat$ingredients_text_col[i], tmp.end.2+end.2-1, tmp.end.2+end.2) == '))' & tot.open != tot.close) {
                  tmp.open.par = 
                    str_count(substr(dat$ingredients_text_col[i], start.2, end.2+tmp.end.2-1),
                              "\\(")
                  tmp.close.par = str_count(substr(dat$ingredients_text_col[i], start.2, end.2+tmp.end.2-1),
                                        "\\)")
                  if(tmp.open.par == tmp.close.par) {
                    end.2 = end.2 + tmp.end.2 -1
                  }
                } else {
                  end.2 = end.2 + tmp.end.2
                }
                
                
              }
              
              # Getting number of open and close parentheses again
              open.par = 
                str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                          "\\(")
              close.par = str_count(substr(dat$ingredients_text_col[i], start.2, end.2),
                                    "\\)")
            } 
          }
          
          ###
          # Don't want the last character if it is a comma or a comma space
          if(max(str_locate(substr(dat$ingredients_text_col[i], end.2, end.2), ","), na.rm = TRUE) > 0) {
            end.2 = end.2 - 1
          } else if (max(str_locate(substr(dat$ingredients_text_col[i], end.2-1, end.2), ",(\\s)"), na.rm = TRUE) > 0) {
            end.2 = end.2 - 2
          }
        }
        
        ###
        # Updating data frame
        # Logic is:
        # If start.1 == start.2, then strings would be the same
        # If start.1 < start.2, then use start.1
        # If start.2 < start.1, then use start.2
        if(!is.na(start.1) & !is.na(start.2) & is.finite(start.1) & is.finite(start.2)) {
          if(start.1 == start.2) {
            if(embedded != 'embedded') {
              # Getting the string embedded in parentheses
              dat[i,paste0('string',j)] = substr(dat$ingredients_text_col[i], start.1, end.1)
              # Getting number of commas before this to indicate which ingredient in the list it is
              dat[i, paste0('ing_string',j)] = str_count(substr(dat$ingredients_text_col[i],1,start.1),",") + 1
            }
            
            # If the first character after the identified end is a close parentheses
            # Then dropping this
            if(substr(dat$ingredients_text_col[i], end.1 + 1, end.1 +1) == ")") {
              dat$ingredients_text_col[i] =
                paste0(substr(dat$ingredients_text_col[i], 1, end.1),
                       substr(dat$ingredients_text_col[i], end.1 + 2, num.char))
            }
            
            # And updating the ingredients text in the data frame
            dat$ingredients_text_col[i] =
              paste0(substr(dat$ingredients_text_col[i], 1, start.1 - 1),
                     substr(dat$ingredients_text_col[i], end.1 + 1, num.char))
          } else if (start.1 < start.2 & start.1 > 1) {
            if(embedded != 'embedded') {
              # Getting the string embedded in parentheses
              dat[i,paste0('string',j)] = substr(dat$ingredients_text_col[i], start.1, end.1)
              # Getting number of commas before this to indicate which ingredient in the list it is
              dat[i, paste0('ing_string',j)] = str_count(substr(dat$ingredients_text_col[i],1,start.1),",") + 1
            }
            
            # If the first character after the identified end is a close parentheses
            # Then dropping this
            if(substr(dat$ingredients_text_col[i], end.1 + 1, end.1 +1) == ")") {
              dat$ingredients_text_col[i] =
                paste0(substr(dat$ingredients_text_col[i], 1, end.1),
                       substr(dat$ingredients_text_col[i], end.1 + 2, num.char))
            }
            
            # And updating the ingredients text in the data frame
            dat$ingredients_text_col[i] =
              paste0(substr(dat$ingredients_text_col[i], 1, start.1 - 1),
                     substr(dat$ingredients_text_col[i], end.1 + 1, num.char))
          } else if (start.2 < start.1 | start.1 == -1) {
            # Getting the string embedded in parentheses
            if(embedded != 'embedded') {
              dat[i,paste0('string',j)] = substr(dat$ingredients_text_col[i], start.2, end.2)
              # Getting number of commas before this to indicate which ingredient in the list it is
              dat[i, paste0('ing_string',j)] = str_count(substr(dat$ingredients_text_col[i],1,start.2),",") + 1
            }
            
            # If the first character after the identified end is a close parentheses
            # Then dropping this
            if(substr(dat$ingredients_text_col[i], end.2 + 1, end.2 +1) == ")") {
              dat$ingredients_text_col[i] =
                paste0(substr(dat$ingredients_text_col[i], 1, end.2),
                       substr(dat$ingredients_text_col[i], end.2 + 2, num.char))
            }
            # And updating the ingredients text in the data frame
            dat$ingredients_text_col[i] =
              paste0(substr(dat$ingredients_text_col[i], 1, start.2 - 1),
                     substr(dat$ingredients_text_col[i], end.2 + 1, num.char))
          } # End of else if statement supdating ingredients text
        } # End of if statement if ingredients text needs to be updated
      } # End of j statement extracted j embedded lists
    } # End of i statement looping through i ingredient lists
    
    # And dropping the column
    dat = dplyr::rename(dat, ingredients_text_parsed = ingredients_text_col)
    
    # Returning data frame
    return(dat)
  } # End of function

#####
###
# Function to correct splits in the parsed ingredient text
# Need to do this because some lists have misplaced parentheses, commas, etc
# i.e. number ( != number )
# This function fixes those issues as best as possible
# Note that this has a lot of exceptions because we're dealing with a huge amount of ingredients!
# There is probably a much much faster way of doing this, but this works for my current purpsoes

corrected.split.function = 
  function(dat,
           cols.loop) {
    for(i in 1:nrow(dat)) {
      tmp.string = dat[i,cols.loop]
      # tmp.string = as.vector(tmp.string[,1:50])
      
      # Getting location of open and close parentheses
      tmp.open = grep("[(]",tmp.string, perl = TRUE)
      tmp.close = grep("[)]",tmp.string, perl = TRUE)
      
      # Getting location of cells where there are not both open and close parentheses
      tmp.open.unique = tmp.open[!(tmp.open %in% tmp.close)]
      tmp.close.unique = tmp.close[!(tmp.close %in% tmp.open)]
      
      # Adding double open and close parentheses
      tmp.double.open = grep("\\(.*\\(",tmp.string)
      tmp.double.close = grep("\\).*\\)",tmp.string)
      
      # Looping to condense ingredient lists
      # If statement - do not need to do this for all items
      if(length(tmp.open.unique) >= 1 &
         length(tmp.open.unique) == length(tmp.close.unique)) {
        # Looping through unique ingredients that have this issue
        for(j in 1:length(tmp.open.unique)) {
          # Pasting ingredients
          tmp.ingredient = paste(tmp.string[1,(tmp.open.unique[j]:tmp.close.unique[j])],
                                 collapse = ",")
          
          # Updating tmp string
          tmp.string[1,tmp.open.unique[j]:tmp.close.unique[j]] <- tmp.ingredient
        }
        
        # And updating list - specifically to remove the ingredients that are part of another ingredient
        # And adding black cells so that list is 50 items long
        tmp.string <- c(unique(as.character(tmp.string)),
                        rep("",50-length(unique(as.character(tmp.string)))))
        
        # And updating cells in data set
        dat[i,cols.loop] <- tmp.string
      }
      
      # Repeating if length of close parentheses is one less than open parentheses
      if(length(tmp.open.unique) >= 1 &
         length(tmp.open.unique) == (length(tmp.close.unique)-1)) {
        # Looping through unique ingredients that have this issue
        # Doing this in a while loop to make an exception for the last entry in the ingredient list
        j = 1
        while(j < length(tmp.open.unique)) { # Updating all but last iteration
          # Pasting ingredients
          tmp.ingredient = paste(tmp.string[1,tmp.open.unique[j]:tmp.close.unique[j]],
                                 collapse = ",")
          
          # Updating tmp string
          tmp.string[1,tmp.open.unique[j]:tmp.close.unique[j]] <- tmp.ingredient
          # And updating counter
          j = j + 1
        }
        while(j == length(tmp.open.unique)) { # Updating last iteration
          # Pasting ingredients
          tmp.ingredient = paste(tmp.string[1,tmp.open.unique[j]:50],
                                 collapse = ",")
          
          # Updating tmp string
          tmp.string[1,tmp.open.unique[j]:50] <- tmp.ingredient
          # And updating counter
          j = j + 1
        }
        # And updating in data frame
        # And updating list - specifically to remove the ingredients that are part of another ingredient
        # And adding black cells so that list is 50 items long
        
        tmp.string <- c(unique(as.character(tmp.string)),
                        rep("",50-length(unique(as.character(tmp.string)))))
        
        # And updating cells in data set
        dat[i,cols.loop] <- tmp.string
      }
      
      # Exception if open parentheses and close parentheses are not within 1 length of each other
      if(length(tmp.open.unique) >= 1 &
         length(tmp.open.unique) > (length(tmp.close.unique)+1)) {
        # Counter
        j = 1
        while(j != length(tmp.open.unique)) { # Repeat for first j-1 instances, where j = length of unique entires with only an open parenthesis
          tmp.ingredient = paste(tmp.string[1,tmp.open.unique[j]:(tmp.open.unique[j+1]-1)],
                                 collapse = ",")
          j = j+1
        }
        if(j == length(tmp.open.unique) &
           max(tmp.close) > max(tmp.open.unique)) { # Repeat for jth instance, if max position of dangling closed parentheses is > max position of dangling open parentheses
          tmp.ingredient = paste(tmp.string[1,tmp.open.unique[j]:(max(tmp.close)-1)],
                                 collapse = ",")
        }
        if(j == length(tmp.open.unique) &
           max(tmp.close) <= max(tmp.open.unique)) { # Repeat for jth instance, if max position of dangling closed parentheses is <= max position of dangling open parentheses
          tmp.ingredient = paste(tmp.string[1,tmp.open.unique[j]:50],
                                 collapse = ",")
        }
      }
    }
    # Returning data frame
    return(dat)
  }

###
# Identifying percent composition of products in a few different ways
# Doing this because layout of numeric text indicating percent within individual ingredients is widely variable
percent.composition.function = 
  function(dat,
           ingredient.col,
           variable.col,
           id.var) {
    
    dat$ingredient_col = dat[,ingredient.col]
    dat$variablen = as.numeric(gsub('V','',dat[,variable.col]))
    dat$id_var = dat[,id.var]
    # Identifying percent composition
    percent0 <-
      (as.vector(str_extract_all(dat$ingredient_col, "[0-9]{1,3}(\\s)?%|[0-9]{1,3}(\\.)[0-9]{1,2}(\\s)?%|[0-9]{1,3}(\\,)[0-9]{1,2}(\\s)?%")))
    
    
    # Updating first time
    morethanonelist = c()
    dat$percent0 = NA
    for(i in 1:nrow(dat)) {
      if(length(percent0[[i]]) == 1) {
        dat$percent0[i] <-
          percent0[[i]][[1]]
      }
      
      if(length(percent0[[i]]) > 1) {
        morethanonelist = c(morethanonelist,i)
      }
    }
    
    # Converting percent numeric
    dat = 
      dat %>%
      mutate(percent_numeric = as.numeric(gsub("%","",percent0))) %>%
      mutate(percent_numeric = ifelse(percent_numeric > 100/variablen,NA,percent_numeric)) %>%
      mutate(percent0 = ifelse(is.na(percent_numeric), NA, percent_numeric))
    
    # Catching commas
    dat$percent0[grep("[0-9]\\,[0-9]",percent0)] <-
      gsub("%","",percent0[grep("[0-9]\\,[0-9]",percent0)]) %>%
      gsub("\\,",".",.) %>% as.numeric()
    
    dat$percent_numeric[grep("[0-9]\\,[0-9]",percent0)] <-
      gsub("%","",percent0[grep("[0-9]\\,[0-9]",percent0)]) %>%
      gsub("\\,",".",.) %>% as.numeric()
    
    # Now going back through to reupdate values with more than one percent identified
    for(i in morethanonelist) {
      # Find min percent value of variables within the same product
      min.value = 
        min(dat$percent0[dat$id_var %in% dat$id_var[i] &
                           dat$variablen < dat$variablen[i]], na.rm = TRUE) %>%
        gsub("%","",.) %>% as.numeric()
      
      # Find max percent value of variables after
      max.value = 
        max(dat$percent0[dat$id_var %in% dat$id_var[i] &
                           dat$variablen > dat$variablen[i]], na.rm = TRUE) %>%
        gsub("%","",.) %>% as.numeric()
      
      # If statements depending on what data is available
      
      # If a percent value is recognized before and after the target ingredient
      if(!is.na(min.value) & !is.na(max.value)) {
        tmp.percent = 
          percent0[[i]] %>% 
          gsub("%","",.) %>% as.numeric(.)
        
        # Finding value which is between neighboring percents
        # And is less than 100/n, where n = location of ingredient in ingredient list
        tmp.percent = 
          tmp.percent[tmp.percent >= max.value &
                        tmp.percent <= min.value &
                        tmp.percent <= 100/dat$variablen[i] &
                        tmp.percent <= (100 - sum(as.numeric(gsub("%","",dat$percent0[dat$id_var == dat$id_var[i]])), na.rm = TRUE))]
        
        if(length(tmp.percent) == 1) {
          dat$percent0[i] = tmp.percent
        } else if (length(tmp.percent) > 1) {
          dat$percent0[i] = max(tmp.percent)
        }
      } else if (!is.na(min.value) & is.na(max.value)) {
        tmp.percent = 
          percent0[[i]] %>% 
          gsub("%","",.) %>% as.numeric(.)
        
        # Finding value which is less than next higher recognized percent
        # And is less than 100/n, where n = location of ingredient in ingredient list
        tmp.percent = 
          tmp.percent[tmp.percent <= min.value &
                        tmp.percent <= 100/dat$variablen[i] &
                        tmp.percent <= (100 - sum(as.numeric(gsub("%","",dat$percent0[dat$id_var == dat$id_var[i]])), na.rm = TRUE))]
        # Updating values
        if(length(tmp.percent) == 1) {
          dat$percent0[i] = tmp.percent
        } else if (length(tmp.percent) > 1) {
          dat$percent0[i] = max(tmp.percent)
        }
      } else if (is.na(min.value) & !is.na(max.value)) {
        tmp.percent = 
          percent0[[i]] %>% 
          gsub("%","",.) %>% as.numeric(.)
        
        # Finding value which is less than next higher recognized percent
        # And is less than 100/n, where n = location of ingredient in ingredient list
        tmp.percent = 
          tmp.percent[tmp.percent >= max.value &
                        tmp.percent <= 100/dat$variablen[i] &
                        tmp.percent <= (100 - sum(as.numeric(gsub("%","",dat$percent0[dat$id_var == dat$id_var[i]])), na.rm = TRUE))]
        # Updating values
        if(length(tmp.percent) == 1) {
          dat$percent0[i] = tmp.percent
        } else if (length(tmp.percent) > 1) {
          dat$percent0[i] = max(tmp.percent)
        }
      } else if (is.na(min.value) & is.na(max.value)) {
        tmp.percent = 
          percent0[[i]] %>% 
          gsub("%","",.) %>% as.numeric(.)
        
        # Finding value which is less than next higher recognized percent
        # And is less than 100/n, where n = location of ingredient in ingredient list
        tmp.percent = 
          tmp.percent[tmp.percent <= 100/dat$variablen[i] &
                        tmp.percent <= (100 - sum(as.numeric(gsub("%","",dat$percent0[dat$id_var == dat$id_var[i]])), na.rm = TRUE))]
        # Updating values
        if(length(tmp.percent) == 1) {
          dat$percent0[i] = tmp.percent
        } else if (length(tmp.percent) > 1) {
          dat$percent0[i] = max(tmp.percent)
        }
      } # End of else if statements based on what data is available
    } # End of loop through ingredients with more than one percent identified
    
    dat <-
      dat %>%
      dplyr::select(-ingredient_col) %>%
      dplyr::select(-id_var)
    
    return(dat)
  }


match.ingredient.function =
  function(dat,
           search.words) {
    # Creating emtpy category
    dat$Food_Category = NA
    
    # rex(search.words$Search_Words[11])
    cols.loop = which(names(search.words) %in% '1') : which(names(search.words) %in% '500')
    
    for(i in 1:nrow(search.words)) {
      # for(i in 1:5) {
      for(j in cols.loop) {
        if(search.words[i,j] != '') {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.words[i,j], ignore.case = TRUE, value) ~ search.words$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      search.words$count[i] <- # Counting foods
        sum(!is.na(dat$Food_Category))
      
      dat$Food_Category = NA # Converting back to NAs
    }
    
    # Reordering based on how common words are
    search.words <- search.words[order(search.words$count),]
    
    # Creating empty data frame to append to
    dat.found = c()
    
    for(i in 1:nrow(search.words)) {
      # for(i in 1:10) {
      for(j in cols.loop) {
        if(search.words[i,j] != '') {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.words[i,j], ignore.case = TRUE, value) ~ search.words$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      # search.words$count[i] <- # Counting foods
      #   sum(!is.na(dat$Food_Category))
      # 
      # dat$Food_Category = NA # Converting back to NAs
    }
    
    ###
    # Manually updating a few values that aren't properly caught
    dat$Food_Category[grepl('tomato',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('tomato',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('white.*choclate',dat$value, ignore.case = TRUE) & grepl('Milk Chocolate|Dark Chocolate',dat$Food_Category)] <- 'Butter, Cream & Ghee'
    dat$Food_Category[grepl('chestnut.*mushroom|mushroom.*chestnut',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mushroom',dat$value, ignore.case = TRUE) & grepl('Crustacean',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mush',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('water',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('oil',dat$value, ignore.case = TRUE) & grepl('Sunflower seeds',dat$Food_Category)] <- 'Sunflower Oil'
    dat$Food_Category[grepl('cocoa|cacao',dat$value, ignore.case = TRUE) & grepl('Butter',dat$Food_Category,ignore.case=TRUE)] <- 'Dark Chocolate'
    dat$Food_Category[grepl('coconut',dat$value, ignore.case = TRUE)] <- 'Other Fruit'
    dat$Food_Category[grepl('coconut.*oil|coconut.*cream',dat$value, ignore.case = TRUE)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value, ignore.case = TRUE) & is.na(dat$Food_Category)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value, ignore.case = TRUE) & grepl('Other Fruit|other veg',dat$Food_Category, ignore.case = TRUE)] <- 'Palm Oil'
    dat$Food_Category_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    dat$Food_Category_sub_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    
    dat$Food_Category[grepl('\\bminimum\\b|\\bmin|min.*fat',dat$value, ignore.case = TRUE) & 
                        is.na(dat$Food_Category) &
                        grepl('butter',dat$product_name,ignore.case = TRUE)] <- 'Butter, Cream & Ghee'
    
    
    return(list(dat,search.words))
    
  }


match.ingredient.function.xxxg.per.100g =
  function(dat,
           search.words) {
    # Creating emtpy category
    dat$Food_Category = NA
    
    # Reordering based on how common words are
    search.words <- search.words[order(search.words$count),]
    
    # Creating empty data frame to append to
    dat.found = c()
    
    # Columns to loop over
    cols.loop = which(names(search.words) %in% '1') : which(names(search.words) %in% '500')
    
    for(i in 1:nrow(search.words)) {
      # for(i in 1:10) {
      for(j in cols.loop) {
        if(search.words[i,j] != '') {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.words[i,j], ignore.case = TRUE, value) ~ search.words$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      # search.words$count[i] <- # Counting foods
      #   sum(!is.na(dat$Food_Category))
      # 
      # dat$Food_Category = NA # Converting back to NAs
    }
    
    ###
    # Manually updating a few values that aren't properly caught
    # So manually updating here
    dat$Food_Category[grepl('tomato',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('tomato',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('white.*choclate',dat$value, ignore.case = TRUE) & grepl('Milk Chocolate|Dark Chocolate',dat$Food_Category)] <- 'Butter, Cream & Ghee'
    dat$Food_Category[grepl('chestnut.*mushroom|mushroom.*chestnut',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mushroom',dat$value, ignore.case = TRUE) & grepl('Crustacean',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mush',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('water',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('oil',dat$value, ignore.case = TRUE) & grepl('Sunflower seeds',dat$Food_Category)] <- 'Sunflower Oil'
    dat$Food_Category[grepl('cocoa|cacao',dat$value, ignore.case = TRUE) & grepl('Butter',dat$Food_Category,ignore.case=TRUE)] <- 'Dark Chocolate'
    dat$Food_Category[grepl('coconut',dat$value, ignore.case = TRUE)] <- 'Other Fruit'
    dat$Food_Category[grepl('coconut.*oil|coconut.*cream',dat$value, ignore.case = TRUE)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value, ignore.case = TRUE) & is.na(dat$Food_Category)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value, ignore.case = TRUE) & grepl('Other Fruit',dat$Food_Category)] <- 'Palm Oil'
    
    dat$Food_Category_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    dat$Food_Category_sub_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    
    dat$Food_Category[grepl('\\bminimum\\b|\\bmin|min.*fat',dat$value, ignore.case = TRUE) & 
                        is.na(dat$Food_Category) &
                        grepl('butter',dat$product_name,ignore.case = TRUE)] <- 'Butter, Cream & Ghee'
    
    return(dat)
    
  }




match.ingredient.function2 =
  function(dat,
           search.words,
           value.col) {
    # Creating emtpy category
    dat$Food_Category = NA
    dat$value.col = dat[,value.col]
    
    # rex(search.words$Search_Words[11])
    cols.loop = which(names(search.words) %in% '1') : which(names(search.words) %in% '500')
    
    for(i in 1:nrow(search.words)) {
      # for(i in 1:5) {
      for(j in cols.loop) {
        if(search.words[i,j] != '') {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.words[i,j], ignore.case = TRUE, value.col) ~ search.words$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      search.words$count[i] <- # Counting foods
        sum(!is.na(dat$Food_Category))
      
      dat$Food_Category = NA # Converting back to NAs
    }
    
    # Reordering based on how common words are
    search.words <- search.words[order(search.words$count),]
    
    # Creating empty data frame to append to
    dat.found = c()
    
    for(i in 1:nrow(search.words)) {
      # for(i in 1:10) {
      for(j in cols.loop) {
        if(search.words[i,j] != '') {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.words[i,j], ignore.case = TRUE, value.col) ~ search.words$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      # search.words$count[i] <- # Counting foods
      #   sum(!is.na(dat$Food_Category))
      # 
      # dat$Food_Category = NA # Converting back to NAs
    }
    
    ###
    # Manually updating a few values that aren't properly caught
    # So manually updating here
    dat$Food_Category[grepl('tomato',dat$value.col, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('tomato',dat$value.col, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Tomatoes'
    dat$Food_Category[grepl('white.*choclate',dat$value.col, ignore.case = TRUE) & grepl('Milk Chocolate|Dark Chocolate',dat$Food_Category)] <- 'Butter, Cream & Ghee'
    dat$Food_Category[grepl('chestnut.*mushroom|mushroom.*chestnut',dat$value, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mushroom',dat$value.col, ignore.case = TRUE) & grepl('Crustacean',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('mush',dat$value.col, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('water',dat$value.col, ignore.case = TRUE) & grepl('Nuts',dat$Food_Category)] <- 'Other Vegetables'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value.col, ignore.case = TRUE) & grepl('Bovine Meat (beef herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('tuna.*steak|fish.*steak',dat$value.col, ignore.case = TRUE) & grepl('Bovine Meat (dairy herd)',dat$Food_Category)] <- 'Fish (farmed)'
    dat$Food_Category[grepl('oil',dat$value.col, ignore.case = TRUE) & grepl('Sunflower seeds',dat$Food_Category)] <- 'Sunflower Oil'
    dat$Food_Category[grepl('cocoa|cacao',dat$value.col, ignore.case = TRUE) & grepl('Butter',dat$Food_Category,ignore.case=TRUE)] <- 'Dark Chocolate'
    dat$Food_Category[grepl('coconut',dat$value.col, ignore.case = TRUE)] <- 'Other Fruit'
    dat$Food_Category[grepl('coconut.*oil|coconut.*cream',dat$value.col, ignore.case = TRUE)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value.col, ignore.case = TRUE) & is.na(dat$Food_Category)] <- 'Palm Oil'
    dat$Food_Category[grepl('\\bpalm',dat$value.col, ignore.case = TRUE) & grepl('Other Fruit|other veg',dat$Food_Category, ignore.case = TRUE)] <- 'Palm Oil'
    
    dat$Food_Category_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    dat$Food_Category_sub_sub[grepl('palm',dat$Food_Category,ignore.case = TRUE)] <- NA
    
    dat$Food_Category[grepl('\\bminimum\\b|\\bmin|min.*fat',dat$value.col, ignore.case = TRUE) & 
                        is.na(dat$Food_Category) &
                        grepl('butter',dat$product_name,ignore.case = TRUE)] <- 'Butter, Cream & Ghee'
    
    
    dat <-
      dat %>% dplyr::select(-value.col)
    
    return(dat)
    
  }


#####
###
# This is the second step - takes the foods as categorised above, and puts them into sub categories when possible
match.ingredient.function.subcategories =
  function(dat,
           search.words.sub,
           embedded) {
   
    # Ordering - don't need to do the count this time
    search.words.sub <- 
      search.words.sub %>%
      arrange(Search_order) %>%
      dplyr::select(LCA_Category, LCA_sub_category, LCA_sub_sub_category,
                    Search_terms, Search_order, Average_of_original_category, max_cat_sub, # Columns for first order of sorting
                    Search_terms_sub, Search_order_sub, Average_of_sub_category, max_cat_sub_sub) %>%
      unique(.)
    
    # Creating empty categories for the filtering
    dat$Food_Category_sub = NA
    dat$Food_Category_sub_sub = NA
    
    if(embedded %in% 'no') {
      tmp.search.words.sub <-
        search.words.sub %>%
        arrange(Search_order) %>%
        dplyr::select(LCA_Category, LCA_sub_category, 
                      Search_terms, Search_order, Average_of_original_category, max_cat_sub) %>%
        unique(.)
      # Looping to assign sub categories
      for(i in 1:nrow(tmp.search.words.sub)) {
        if(tmp.search.words.sub$Search_order[i] != tmp.search.words.sub$max_cat_sub[i]) { # Assign to sub categories based on search terms
          dat <-
            dat %>%
            mutate(Food_Category_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub$LCA_Category[i] &
                              grepl(tmp.search.words.sub$Search_terms[i], value, ignore.case = TRUE),
                            tmp.search.words.sub$LCA_sub_category[i], Food_Category_sub)) 
        } else { # Assign all remaining values to the sub category
          dat <-
            dat %>%
            mutate(Food_Category_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub$LCA_Category[i] &
                              is.na(Food_Category_sub),
                            tmp.search.words.sub$LCA_sub_category[i], Food_Category_sub))
        } # End of if else statement
      } # End of loop through rows in the search terms file
      
      # Now going from the sub categories to the sub sub categories
      # E.g. the above loop does brassicas -> cauliflower and broccoli
      # The below does cauliflower and broccoli -> broccoli
      tmp.search.words.sub.sub <-
        search.words.sub %>%
        filter(LCA_sub_sub_category != '') %>%
        arrange(Search_order_sub) %>%
        dplyr::select(LCA_Category, LCA_sub_category, LCA_sub_sub_category, 
                      Search_terms_sub, Search_order_sub, Average_of_sub_category, max_cat_sub_sub) %>%
        unique(.)
      # Looping through the sub categories
      for(i in 1:nrow(tmp.search.words.sub.sub)) {
        if(tmp.search.words.sub.sub$Search_order_sub[i] != tmp.search.words.sub.sub$max_cat_sub_sub[i]) { # Assign to sub categories based on search terms
          dat <-
            dat %>%
            mutate(Food_Category_sub_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub.sub$LCA_Category[i] &
                              Food_Category_sub %in% tmp.search.words.sub.sub$LCA_sub_category[i] &
                              grepl(tmp.search.words.sub.sub$Search_terms_sub[i], value, ignore.case = TRUE),
                              tmp.search.words.sub.sub$LCA_sub_sub_category[i], Food_Category_sub_sub)) 
        } else { # Assign all remaining values to the sub category
          dat <-
            dat %>%
            mutate(Food_Category_sub_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub.sub$LCA_Category[i] &
                              Food_Category_sub %in% tmp.search.words.sub.sub$LCA_sub_category[i] &
                              is.na(Food_Category_sub_sub),
                            tmp.search.words.sub.sub$LCA_sub_sub_category[i], Food_Category_sub_sub))
        } # End of if else statement
      } # End of loop through rows in the search terms file
      
      
    } else if (embedded %in% 'yes') { # Function for embedded ingredients - same as above, just different column names
      tmp.search.words.sub <-
        search.words.sub %>%
        arrange(Search_order) %>%
        dplyr::select(LCA_Category, LCA_sub_category, 
                      Search_terms, Search_order, Average_of_original_category, max_cat_sub) %>%
        unique(.)
      # Looping to assign sub categories
      for(i in 1:nrow(tmp.search.words.sub)) {
        if(tmp.search.words.sub$Search_order[i] != tmp.search.words.sub$max_cat[i]) { # Assign to sub categories based on search terms
          dat <-
            dat %>%
            mutate(Food_Category_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub$LCA_Category[i] &
                              grepl(tmp.search.words.sub$Search_terms[i], value_embedded, ignore.case = TRUE),
                            tmp.search.words.sub$LCA_sub_category[i], Food_Category_sub)) 
        } else { # Assign all remaining values to the sub category
          dat <-
            dat %>%
            mutate(Food_Category_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub$LCA_Category[i] &
                              is.na(Food_Category_sub),
                            tmp.search.words.sub$LCA_sub_category[i], Food_Category_sub))
        } # End of if else statement
      } # End of loop through rows in the search terms file
      
      # Now going from the sub categories to the sub sub categories
      # E.g. the above loop does brassicas -> cauliflower and broccoli
      # The below does cauliflower and broccoli -> broccoli
      tmp.search.words.sub.sub <-
        search.words.sub %>%
        filter(LCA_sub_sub_category != '') %>%
        arrange(Search_order_sub) %>%
        dplyr::select(LCA_Category, LCA_sub_category, LCA_sub_sub_category, 
                      Search_terms_sub, Search_order_sub, Average_of_sub_category, max_cat_sub_sub) %>%
        unique(.)
      # Looping through the sub categories
      for(i in 1:nrow(tmp.search.words.sub.sub)) {
        if(tmp.search.words.sub.sub$Search_order_sub[i] != tmp.search.words.sub.sub$max_cat_sub_sub[i]) { # Assign to sub categories based on search terms
          dat <-
            dat %>%
            mutate(Food_Category_sub_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub.sub$LCA_Category[i] &
                              Food_Category_sub %in% tmp.search.words.sub.sub$LCA_sub_category[i] &
                              grepl(tmp.search.words.sub.sub$Search_terms_sub[i], value_embedded, ignore.case = TRUE),
                            tmp.search.words.sub.sub$LCA_sub_sub_category[i], Food_Category_sub_sub)) 
        } else { # Assign all remaining values to the sub category
          dat <-
            dat %>%
            mutate(Food_Category_sub_sub = 
                     ifelse(Food_Category %in% tmp.search.words.sub.sub$LCA_Category[i] &
                              Food_Category_sub %in% tmp.search.words.sub.sub$LCA_sub_category[i] &
                              is.na(Food_Category_sub_sub),
                            tmp.search.words.sub.sub$LCA_sub_sub_category[i], Food_Category_sub_sub))
        } # End of if else statement
      } # End of loop through rows in the search terms file
      
    }
    
    return(dat)
  }



#####
###
# Function to interpolate % composition of food ingredients
interpolate.food.ingredients <-
  function(dat) {
    
    # list of variables to loop through
    variable.list = paste0('V',1:length(unique(dat$variable)))
    
    # Some data management
    dat <- 
      dat %>% 
      mutate(percent = percent0) %>%
      mutate(Aisle_Shelf = paste0(Aisle, "_", Shelf)) %>%
      mutate(percent_estimated = NA)
    
    # Saving this to avoid updating averages in the department/aisle/shelf based on already updated averages
    dat.keep <- dat
    
    # Logical sense check for percentages
    for(i in variable.list) { # percent composition cannot be greater than 100/n, where n = order of ingredient in the product
      dat$percent[dat$variable == i & dat$percent > 100/which(variable.list == i) & !is.na(dat$percent)] <- NA
    }
    
    # Creating new column with identified percent composition
    # Identifies rows where percent was listed in back of package information
    dat <-
      dat %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    # Updating values based on shelf first, if more than 10 products in that shelf
    # But only if NAs
    shelf.sum <-
      dat.keep %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Shelf, Food_Category, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                count = sum(count)) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shelf.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_shelf, percent)) %>%
      dplyr::select(-percent_shelf)
    
    # Updating values based on aisle, if more than 10 products in that aisle
    aisle.sum <-
      dat.keep %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Food_Category, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                count = sum(count)) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable)), NA, percent_aisle))
    
    dat <-
      left_join(dat, aisle.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_aisle, percent)) %>%
      dplyr::select(-percent_aisle)
    
    # Updating values based on department
    department.sum <-
      dat.keep %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, Food_Category, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department))
    
    dat <-
      left_join(dat, department.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_department, percent)) %>%
      dplyr::select(-percent_department)
    
    # Repeating the above across all categories
    # Doing this by building regression models for each Aisle and Shelf
    # Structure of model is percent = 1/variable^2, w/ no intercept to avoid negative values
    # Where variable is the location of variable in the product's ingredient list

    # Starting with each shelf
    # List to loop
    loop.list <- unique(dat.keep$Aisle_Shelf)
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        filter(Aisle_Shelf %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z])
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z]) > 5) {
        tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        # Only updating if adj R^2 > .5 and P < .05
        if(summary(tmp.lm)$adj.r.squared > 0.5 & summary(tmp.lm)$coefficients[4] < 0.05) {
          dat <-
            dat %>%
            mutate(percent_estimated = ifelse(Aisle_Shelf %in% z,
                                              tmp.lm$coefficients[[1]] / variablen^2,
                                              percent_estimated))
          
        }
      }
    }
    
    # And udpating values for shelves
    dat <-
      dat %>%
      mutate(percent = ifelse(!is.na(percent_estimated) & is.na(percent),
                              percent_estimated,
                              percent))
    
    # Repeating for Aisles
    # Setting percent_estimated back to 0
    dat <-
      dat %>%
      mutate(percent_estimated = NA)
    # List to loop
    loop.list <- unique(dat.keep$Aisle)
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        filter(Aisle %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle %in% z])
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle %in% z]) > 5) {
        tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        # Only updating if adj R^2 > .5 and P < .05
        if(summary(tmp.lm)$adj.r.squared > 0.5 & summary(tmp.lm)$coefficients[4] < 0.05) {
          dat <-
            dat %>%
            mutate(percent_estimated = ifelse(Aisle %in% z,
                                              tmp.lm$coefficients[[1]] / variablen^2,
                                              percent_estimated))
          
        }
      }
    }
    
    # And udpating values based on regression for Aisles
    dat <-
      dat %>%
      mutate(percent = ifelse(!is.na(percent_estimated) & is.na(percent),
                              percent_estimated,
                              percent)) %>%
      dplyr::select(-percent_estimated) %>% # And dropping variable
      dplyr::select(-Aisle_Shelf) # And dropping Aisle_Shelf indicator
    
    # Repeating for Departments
    # Don't need a loop, because this already loops across departments
    
    
    # Repeating for departments
    dep.sum <- 
      dat.keep %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      mutate(percent_department = ifelse(count < 10, NA, percent_department)) %>% 
      dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department)) %>%
      mutate(variablen = as.numeric(gsub("V","",variable)))
    
    # Building regression to estimate composition of other ingredients
    if(prop_variables >= 0.5 & max(dat.keep$variablen) > 5) {
      tmp.lm = lm(dat.keep$percent ~ I(1/dat.keep$variablen^2) - 1) # No intercept
      # Predicting
      dep.sum <-
        dep.sum %>%
        mutate(percent_estimated = tmp.lm$coefficients[[1]] / variablen^2) %>% # Predicting 
        mutate(percent_department = ifelse(is.na(percent_department), percent_estimated, percent_department)) %>%
        dplyr::select(variable, percent_department)
      # Merging back in
      dat <-
        left_join(dat, dep.sum) %>%
        mutate(percent = ifelse(is.na(percent), percent_department, percent)) %>%
        dplyr::select(-percent_department)
    }
    
    
    
    
    # Now updating values based on what is known
    # Getting list of products to interpolate (i.e. any product where total percent != 100) and n ingredients > 1
    product.percent <-
      dat %>%
      group_by(id.new, product_name) %>%
      dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
    
    # merging in
    # And setting order of ingredients in product
    dat <- 
      left_join(dat, product.percent) %>%
      transform(variable = factor(variable, levels = variable.list)) %>%
      mutate(tot_percent = tot_percent_new) %>%
      dplyr::select(-tot_percent_new)
    
    # Ordering
    dat <-
      dat[order(dat$id.new,dat$variable),]
    
    # And list of products to loop through
    product.list <- 
      product.percent$id.new[product.percent$tot_percent_new != 100]
    
    # Looping through these products to update percent composition
    # This is a behemoth of a loop
    for(j in product.list) {
      # Temporary data frame
      tmp.dat <- dat[dat$id.new %in% j,]
      
      # Easy exception
      # If only one "NO" in keep_percent, then that needs to change such that total percent sums to 100
      if(sum(tmp.dat$keep_percent == 'NO') == 1) {
        tmp.dat$percent[tmp.dat$keep_percent == 'NO'] <-
          100 - sum(tmp.dat$percent[tmp.dat$keep_percent == 'YES'])
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
      } else if(sum(is.na(tmp.dat$percent)) == nrow(tmp.dat))  {
        tmp.dat$percent = 100/nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (sum(tmp.dat$percent, na.rm = TRUE) == 0) {
        tmp.dat$percent = 100 / nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (nrow(tmp.dat) == 1) {
        tmp.dat$percent = 100
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else {
        # Logic here is to interpolate other values
        # Getting vector of known percents
        tmp.vector <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
        # Get total percent composition of tmp vector
        tot.percent <- sum(tmp.vector, na.rm = TRUE)
        
        ###
        # Interpolate normally
        # If sum tmp vector = 0, then cannot do anything
        # Therefore, if sum does not equal 0
        if(sum(tmp.vector, na.rm = TRUE) != 0) {
          tmp.vector1 = tmp.vector
          
          # Logic checks to fix percent composition of ingredients
          # (1) For example, if N2 == N4, then N3 = N4. Flip tmp.keep to "KEEP"
          # Doing this for every instance this appears
          if(length(tmp.vector1[tmp.keep %in% 'YES']) != length(unique(tmp.vector1[tmp.keep %in% 'YES']))) {
            # Finding location of known composition that are equal
            tmp.table <- 
              table(tmp.vector1[tmp.keep %in% 'YES']) %>%
              .[. != 1] %>% names(.) %>% as.numeric(.)
            
            for(x in 1:length(tmp.table)) {
              tmp.indices <- which(tmp.vector1 == tmp.table[x] & tmp.keep %in% 'YES')
              if(length(tmp.indices) > 0) {
                tmp.vector1[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <-
                  tmp.vector1[tmp.indices[1]]
                tmp.keep[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <- "YES"
              }
            }
            
          }
          

          # Getting indices with reported percents
          tmp.check <- which(tmp.vector != 0)
          
          # But can only interpolate if more than 1 ingredient is recognized with a percent
          # if(length(tmp.check) > 1) {
          if(length(tmp.check) > 1) {
            for(k in 1:(length(tmp.check) - 1)) {
              if(!(k %in% c(0))) {
                tmp.vector1[tmp.check[k]:tmp.check[k+1]] <-
                  seq(from = tmp.vector1[tmp.check[k]], to = tmp.vector1[tmp.check[k+1]], length.out = (tmp.check[k+1]-tmp.check[k] + 1)) 
              }
            }
          }
          
          
          # # Updating ingredients for first listed ingredient
          # if(min(tmp.check) != 1) {
          #   tmp.vector1[1:(min(tmp.check)-1)] <-
          #     tmp.vector1[min(tmp.check)]
          # }
        }
        
        ###
        # And exception if total sum of all ingredients > 100 when interpolating normally
        # Only doing this for the last gap in the ingredients list
        # E.g. if ingredients 5-9 are unknown, interpolate these normally
        # But updated interpolation for ingredients 1-3
        # Doing it this way because there is normally a big dropoff in percent composition after the first ingredient or two
        
        
        
        # if(sum(tmp.vector1) > 100) { # If true, then total composition of all ingredients recognized is greater than 100
        #   # Only updating the first gap in the ingredient list
        #   # Assuming that the other graps have been properly managed
        #   
        #   # Removing values that were identified
        #   tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 0
        #   
        #   # Getting sum composition of the other products
        #   # Remaining ingredients need to make this sum to 100
        #   tmp.sum <- sum(tmp.vector1)
        #   
        #   # If only 1 missing ingredient, then all of the missing values are assigned to this
        #   if(length((tmp.check[1] + 1) : (tmp.check[2] -1)) == 1) {
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 100 - tmp.sum
        #   } else { # if more than 1 missing ingredient
        #     # Getting what the interpolation would be
        #     tmp.int <- seq(from = tmp.vector1[tmp.check[1]], to = tmp.vector1[tmp.check[2]], length.out = (tmp.check[2] - tmp.check[1]) + 1)
        #     # Getting only the 2nd to nth values
        #     tmp.int <- tmp.int[2:(length(tmp.int)-1)]
        #     # Now weighting missing ingredients by the values they would be if interpolation was correct
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- # Index of missing values
        #       (100-tmp.sum) * # Total missing percent
        #       tmp.int / # This is what the interpolation would be if the total ingredient list didn't sum to 100
        #       sum(tmp.int) # Then dividing by sum of this interpolation to get percent of missing ingredients
        #   }
        # }
        
        
        ###
        # Updating percentages of last n ingredients after the last ingredient that has a percent recognized
        if(max(tmp.check) < nrow(tmp.dat)) {
          
          tmp.keep[length(tmp.keep)] <- 'YES'
          # Getting how these would interpolate, assuming the 10th ingredient is .1%
          tmp.int <- seq(from = tmp.vector1[tmp.check[length(tmp.check)]], # Value of the last recognized ingredient
                         to = min(c(.1,tmp.vector1[tmp.check[length(tmp.check)]])), # Assuming that the last ingredient is the lesser of 1% or last identified % composition
                         length.out = (nrow(tmp.dat)-tmp.check[length(tmp.check)] + 1)) # Number of ingredients needed
          
          if(length(tmp.int) >= 1) { # Logic check in case there aren't any more ingredients to identify
            # Updating tmp.vector
            tmp.vector1[(tmp.check[length(tmp.check)] + 1) : (tmp.check[length(tmp.check)] + length(tmp.int) - 1)] <-
              tmp.int[2:length(tmp.int)]
          }
        }
        
        
        
        ###
        # Updating percentages or first n ingredients if only one percentage is recognized
        if(min(tmp.check) != 1) { # Logic statement to identify products where the first identified ingredient is not the first
          tmp.sum <- sum(tmp.vector1, na.rm = TRUE) # total composition of ingredients
          tmp.int <- seq(from = (100-tmp.sum), # Percent not yet recognized
                         to = tmp.vector1[min(tmp.check)], # Percent composition of the first product
                         length.out = (min(tmp.check)))
          if(length(tmp.int)>=1) {
            tmp.vector1[1:(min(tmp.check))] <- tmp.int
          }
        }
        
        ###
        # Looping through to correct values where i.e. v1 < v2, etc
        # But only adjusting the non-keep percentages
        for(z in (length(tmp.vector1) - 1):1) {
          if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
            tmp.vector1[z] <- tmp.vector1[z+1]
          }
        }
        
        ###
        # Assuming the last ingredient is correct
        # Need to do this to provide a counterpoint for everything else
        # And also serves as a basis to solve the rest of this puzzle
        tmp.keep[length(tmp.keep)] <- 'YES'
        
        ###
        # And now correcting percent composition when they do not sum to 100
        # Doing this in a series of while loops
        # Where logic is
        # (a) Adjust the "non-keep" ingredients such that everything sums to 100
        # (b) check that e.g. v1 >= v2, etc
        # (c) check that e.g. v2 >= 100/2
        counter = 1
        while(sum(tmp.vector1) != 100 & counter <= 10) {
          counter = counter + 1
          if(round(sum(tmp.vector1)) != 100) {
            # Getting sum percent of the interpolated ingredients
            tmp.sum.int <- sum(tmp.vector1[tmp.keep == 'NO'])
            tmp.sum <- sum(tmp.vector1[tmp.keep == 'YES'])
            # And scaling so that all ingredients sum to 100
            tmp.vector1[tmp.keep == 'NO'] <-
              tmp.vector1[tmp.keep == 'NO'] *
              ((100 - tmp.sum) / tmp.sum.int)
            
            tmp.vector1[is.nan(tmp.vector1)] <- 0
          } # End if statement
          
          ###
          # Doing two few checks, specifically 
          # (1) %1 > %2 > %3 etc
          # Looping through to correct values where i.e. v1 < v2, etc
          # But only adjusting the non-keep percentages
          for(z in (length(tmp.vector1) - 1):1) {
            if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
              tmp.vector1[z] <- tmp.vector1[z+1]
            }
          }
          
          
          # (2)
          # Updating keep percent when possible
          for(z in (length(tmp.vector1) - 1) : 1) {
            if(tmp.vector1[z] == tmp.vector1[z+1] &
               tmp.keep[z+1] == 'YES') {
              tmp.keep[z] <- 'YES'
            }
          }

          # (3) %n msut be less than 100/n
          # But only adjusting the non-identified ingredients
          # And then flagging ones that cannot change again
          for(i in length(tmp.vector1):2) {
            if(tmp.vector1[i] > 100/i & tmp.keep == 'NO') {
              tmp.vector1[i] <- 100/i
            } # End for loop for ingredients
          }

        } # End while loop
      } # End of big if else loop
      
      
      
      # updating values in the data frame
      dat$percent[dat$id.new %in% j] <-
        tmp.vector1
      dat$tot_percent[dat$id.new %in% j] <- sum(tmp.vector1, na.rm = TRUE)
    } # End of loop for products
    
    # Dropping id.new
    # dat <- dat %>% dplyr::select(-id.new)
    # Returning data frame
    return(dat)
  } # End function for ingredient interpolating



# Function to interpolate percent for embedded ingredients

interpolate.food.ingredients.embedded <-
  function(dat) {
    
    # list of variables to loop through
    variable.list = paste0('V',1:length(unique(dat$variable_embedded)))
    
    # Some data management
    dat <- 
      dat %>% 
      mutate(percent = percent0) %>%
      transform(variable_embedded = factor(variable_embedded, levels = variable.list))
    
    # Logical sense check for percentages
    for(i in variable.list) { # percent composition cannot be greater than 100/n, where n = order of ingredient in the product
      dat$percent[dat$variable == i & dat$percent > 100/which(variable.list == i) & !is.na(dat$percent)] <- NA
    }
    
    # Creating new column with identified percent composition
    # Identifies rows where percent was listed in back of package information
    dat <-
      dat %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    # Updating values based on shelf first, if more than 10 products in that shelf
    shelf.sum <-
      dat.ingredients.nuts %>%
      filter(count_comma_embedded == 0) %>%
      group_by(Aisle, Shelf, Food_Category, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent_numeric, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Shelf != 'NULL') %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shelf.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_shelf, percent)) %>%
      dplyr::select(-percent_shelf)
    
    # Updating values based on aisle, if more than 10 products in that aisle
    aisle.sum <-
      dat.ingredients.nuts %>%
      filter(count_comma_embedded == 0) %>%
      group_by(Aisle, Food_Category, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent_numeric, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Aisle != 'NULL') %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_aisle))
    
    dat <-
      left_join(dat, aisle.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_aisle, percent)) %>%
      dplyr::select(-percent_aisle)
    
    # Updating values based on department
    department.sum <-
      dat.ingredients.nuts %>%
      filter(count_comma_embedded == 0) %>%
      group_by(Department, Food_Category, variable) %>%
      dplyr::summarise(percent_department = mean(percent_numeric, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Department != 'NULL') %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_department))
    
    dat <-
      left_join(dat, department.sum %>% filter(!is.na(Food_Category))) %>%
      mutate(percent = ifelse(is.na(percent), percent_department, percent)) %>%
      dplyr::select(-percent_department)
    
    # Repeating the above across all categories
    shel.sum <-
      dat.ingredients.nuts %>%
      filter(count_comma_embedded == 0) %>%
      group_by(Aisle, Shelf, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent_numeric, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Shelf != 'NULL') %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shel.sum) %>%
      mutate(percent = ifelse(is.na(percent), percent_shelf, percent)) %>%
      dplyr::select(-percent_shelf)
    
    
    ais.sum <-
      dat.ingredients.nuts %>%
      group_by(Aisle, variable) %>%
      filter(count_comma_embedded == 0) %>%
      dplyr::summarise(percent_aisle = mean(percent_numeric, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Aisle != 'NULL') %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_aisle))
    
    dat <-
      left_join(dat, ais.sum) %>%
      mutate(percent = ifelse(is.na(percent), percent_aisle, percent)) %>%
      dplyr::select(-percent_aisle)
    
    dep.sum <- 
      dat.ingredients.nuts %>%
      group_by(Department, variable) %>%
      filter(count_comma_embedded == 0) %>%
      dplyr::summarise(percent_department = mean(percent_numeric, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>%
      mutate(variable_embedded = variable) %>% dplyr::select(-variable) %>% filter(Department != 'NULL') %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable_embedded)), NA, percent_department))
    
    dat <-
      left_join(dat, dep.sum) %>%
      mutate(percent = ifelse(is.na(percent), percent_department, percent)) %>%
      dplyr::select(-percent_department)
    
    
    # Now updating values based on what is known
    # Getting list of products to interpolate (i.e. any product where total percent != 100) and n ingredients > 1
    product.percent <-
      dat %>%
      group_by(id_embedded, product_name, variable) %>%
      dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
    
    # merging in
    # And setting order of ingredients in product
    dat <- 
      left_join(dat, product.percent) %>%
      transform(variable_embedded = factor(variable_embedded, levels = variable.list)) %>%
      mutate(tot_percent = tot_percent_new) %>%
      dplyr::select(-tot_percent_new)
    
    # Ordering
    dat <-
      dat[order(dat$id_embedded,dat$variable_embedded),]
    
    # Creating temporary column to go through products
    # dat <-
    #   dat %>%
    #   mutate(id.new = paste0(id, variable, product_name, Department, Aisle, Shelf))
    
    ###
    # Updating percent for variables with 1 ingredient
    dat.count <-
      dat %>%
      group_by(id_embedded) %>%
      dplyr::summarise(count = dplyr::n()) %>%
      filter(count == 1)
    
    dat <-
      dat %>%
      mutate(percent = ifelse(id_embedded %in% dat.count$id_embedded, 100, percent)) %>%
      mutate(tot_percent = ifelse(id_embedded %in% dat.count$id_embedded, 100, tot_percent))
    
    # Updating percent for ingredients with no percents identified
    dat.no.count <- 
      dat %>%
      mutate(count_no = ifelse(is.na(percent),1,0)) %>%
      group_by(id_embedded) %>%
      dplyr::summarise(count = sum(count_no))
    
    # And doing again
    product.percent <-
      dat %>%
      group_by(id_embedded, product_name, variable) %>%
      dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
    
    # And list of products to loop through
    product.list <- 
      product.percent$id_embedded[product.percent$tot_percent_new != 100]
    
    # Looping through these products to update percent composition
    # This is a behemoth of a loop
    for(j in product.list) {
      # Temporary data frame
      tmp.dat <- dat[dat$id_embedded %in% j,]
      
      # Easy exception
      # If only one "NO" in keep_percent, then that needs to change such that total percent sums to 100
      if(sum(tmp.dat$keep_percent == 'NO') == 1) {
        tmp.dat$percent[tmp.dat$keep_percent == 'NO'] <-
          100 - unique(tmp.dat$tot_percent)
        tmp.dat$keep_percent <- 'YES'
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (nrow(tmp.dat) == dat.no.count$count[dat.no.count$id_embedded %in% j]) {
        tmp.dat$percent <- 100/nrow(tmp.dat)
        tmp.dat$tot_percent <- 100
        tmp.dat$keep_percent <- 'YES'
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (sum(tmp.dat$percent, na.rm = TRUE) == 0) {
        tmp.dat$percent = 100/nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (nrow(tmp.dat) == 1) {
        tmp.dat$percent = 100
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else {
        # Logic here is to interpolate other values
        # Getting vector of known percents
        tmp.vector <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
        # Get total percent composition of tmp vector
        tot.percent <- sum(tmp.vector, na.rm = TRUE)
        
        ###
        # Interpolate normally
        # If sum tmp vector = 0, then cannot do anything
        # Therefore, if sum does not equal 0
        if(sum(tmp.vector, na.rm = TRUE) != 0) {
          tmp.vector1 = tmp.vector
          # Getting indices with reported percents
          tmp.check <- which(tmp.vector != 0)
          
          # Logic checks to fix percent composition of ingredients
          # (1) For example, if N2 == N4, then N3 = N4. Flip tmp.keep to "KEEP"
          # Doing this for every instance this appears
          if(length(tmp.vector1[tmp.keep %in% 'YES']) != length(unique(tmp.vector1[tmp.keep %in% 'YES']))) {
            # Finding location of known composition that are equal
            tmp.table <- 
              table(tmp.vector1[tmp.keep %in% 'YES']) %>%
              .[. != 1] %>% names(.) %>% as.numeric(.)
            
            for(x in 1:length(tmp.table)) {
              tmp.indices <- which(tmp.vector1 == tmp.table[x] & tmp.keep %in% 'YES')
              if(length(tmp.indices) > 0) {
                tmp.vector1[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <-
                  tmp.vector1[tmp.indices[1]]
                tmp.keep[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <- "YES"
              }
            }
            
          }
          
          # But can only interpolate if more than 1 ingredient is recognized with a percent
          # if(length(tmp.check) > 1) {
          if(length(tmp.check) > 1) {
            for(k in 1:(length(tmp.check) - 1)) {
              if(!(k %in% c(0))) {
                tmp.vector1[tmp.check[k]:tmp.check[k+1]] <-
                  seq(from = tmp.vector1[tmp.check[k]], to = tmp.vector1[tmp.check[k+1]], length.out = (tmp.check[k+1]-tmp.check[k] + 1)) 
              }
            }
          }
          
          
          # # Updating ingredients for first listed ingredient
          # if(min(tmp.check) != 1) {
          #   tmp.vector1[1:(min(tmp.check)-1)] <-
          #     tmp.vector1[min(tmp.check)]
          # }
        }
        
        ###
        # And exception if total sum of all ingredients > 100 when interpolating normally
        # Only doing this for the last gap in the ingredients list
        # E.g. if ingredients 5-9 are unknown, interpolate these normally
        # But updated interpolation for ingredients 1-3
        # Doing it this way because there is normally a big dropoff in percent composition after the first ingredient or two
        
        
        
        # if(sum(tmp.vector1) > 100) { # If true, then total composition of all ingredients recognized is greater than 100
        #   # Only updating the first gap in the ingredient list
        #   # Assuming that the other graps have been properly managed
        #   
        #   # Removing values that were identified
        #   tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 0
        #   
        #   # Getting sum composition of the other products
        #   # Remaining ingredients need to make this sum to 100
        #   tmp.sum <- sum(tmp.vector1)
        #   
        #   # If only 1 missing ingredient, then all of the missing values are assigned to this
        #   if(length((tmp.check[1] + 1) : (tmp.check[2] -1)) == 1) {
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 100 - tmp.sum
        #   } else { # if more than 1 missing ingredient
        #     # Getting what the interpolation would be
        #     tmp.int <- seq(from = tmp.vector1[tmp.check[1]], to = tmp.vector1[tmp.check[2]], length.out = (tmp.check[2] - tmp.check[1]) + 1)
        #     # Getting only the 2nd to nth values
        #     tmp.int <- tmp.int[2:(length(tmp.int)-1)]
        #     # Now weighting missing ingredients by the values they would be if interpolation was correct
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- # Index of missing values
        #       (100-tmp.sum) * # Total missing percent
        #       tmp.int / # This is what the interpolation would be if the total ingredient list didn't sum to 100
        #       sum(tmp.int) # Then dividing by sum of this interpolation to get percent of missing ingredients
        #   }
        # }
        
        
        ###
        # Updating percentages of last n ingredients after the last ingredient that has a percent recognized
        if(max(tmp.check) < nrow(tmp.dat)) {
          tmp.keep[length(tmp.keep)] <- 'YES'
          # Getting how these would interpolate, assuming the 10th ingredient is 1%
          tmp.int <- seq(from = tmp.vector1[tmp.check[length(tmp.check)]], # Value of the last recognized ingredient
                         to = min(c(1,tmp.vector1[tmp.check[length(tmp.check)]])), # Assuming that the last ingredient is the smaller of 1% of the total or the minimum of the recognized ingredients 
                         length.out = (nrow(tmp.dat)-tmp.check[length(tmp.check)] + 1)) # Number of ingredients needed
          
          if(length(tmp.int) >= 1) { # Logic check in case there aren't any more ingredients to identify
            # Updating tmp.vector
            tmp.vector1[(tmp.check[length(tmp.check)] + 1) : (tmp.check[length(tmp.check)] + length(tmp.int) - 1)] <-
              tmp.int[2:length(tmp.int)]
          }
        }
        
        
        
        ###
        # Updating percentages or first n ingredients if only one percentage is recognized
        if(min(tmp.check) != 1) { # Logic statement to identify products where the first identified ingredient is not the first
          tmp.sum <- sum(tmp.vector1, na.rm = TRUE) # total composition of ingredients
          tmp.int <- seq(from = (100-tmp.sum), # Percent not yet recognized
                         to = tmp.vector1[min(tmp.check)], # Percent composition of the first product
                         length.out = (min(tmp.check)))
          if(length(tmp.int)>=1) {
            tmp.vector1[1:(min(tmp.check))] <- tmp.int
          }
        }
        
        ###
        # Looping through to correct values where i.e. v1 < v2, etc
        # But only adjusting the non-keep percentages
        for(z in (length(tmp.vector1) - 1):1) {
          if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
            tmp.vector1[z] <- tmp.vector1[z+1]
          }
        }
        
        ###
        # Assuming the last ingredient is correct
        # Need to do this to provide a counterpoint for everything else
        # And also serves as a basis to solve the rest of this puzzle
        tmp.keep[length(tmp.keep)] <- 'YES'
        
        ###
        # And now correcting percent composition when they do not sum to 100
        # Doing this in a series of while loops
        # Where logic is
        # (a) Adjust the "non-keep" ingredients such that everything sums to 100
        # (b) check that e.g. v1 >= v2, etc
        # (c) check that e.g. v2 >= 100/2
        counter = 1
        while(sum(tmp.vector1) != 100 & counter <= 10) {
          counter = counter + 1
          if(round(sum(tmp.vector1)) != 100) {
            # Getting sum percent of the interpolated ingredients
            tmp.sum.int <- sum(tmp.vector1[tmp.keep == 'NO'])
            tmp.sum <- sum(tmp.vector1[tmp.keep == 'YES'])
            # And scaling so that all ingredients sum to 100
            tmp.vector1[tmp.keep == 'NO'] <-
              tmp.vector1[tmp.keep == 'NO'] *
              ((100 - tmp.sum) / tmp.sum.int)
            
            tmp.vector1[is.nan(tmp.vector1)] <- 0
          } # End if statement
          
          ###
          # Doing two few checks, specifically 
          # (1) %1 > %2 > %3 etc
          # Looping through to correct values where i.e. v1 < v2, etc
          # But only adjusting the non-keep percentages
          for(z in (length(tmp.vector1) - 1):1) {
            if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
              tmp.vector1[z] <- tmp.vector1[z+1]
            }
          }
          
          
          # (2)
          # Updating keep percent when possible
          for(z in (length(tmp.vector1) - 1) : 1) {
            if(tmp.vector1[z] == tmp.vector1[z+1] &
               tmp.keep[z+1] == 'YES') {
              tmp.keep[z] <- 'YES'
            }
          }
          
          # (3) %n msut be less than 100/n
          # But only adjusting the non-identified ingredients
          # And then flagging ones that cannot change again
          for(i in length(tmp.vector1):2) {
            if(tmp.vector1[i] > 100/i & tmp.keep[i] == 'NO') {
              tmp.vector1[i] <- 100/i
            } # End for loop for ingredients
          }
          
        } # End while loop
      }
      
      
      
      # updating values in the data frame
      dat$percent[dat$id_embedded %in% j] <-
        tmp.vector1
      dat$tot_percent[dat$id_embedded %in% j] <- sum(tmp.vector1, na.rm = TRUE)
    } # End of loop for products
    
    # Returning data frame
    return(dat)
  } # End function for ingredient interpolating




###
# Function to estimate env impacts of different food products
env.estimate.function =
  function(dat,
           food.col.dat,
           percent.col,
           tot.percent.col,
           lca.dat,
           food.col.lcadat) {
    
    # Creating columns to integrate with function
    # These will be dropped later
    dat$tot.percent.col = dat[,tot.percent.col]
    dat$food.col = dat[,food.col.dat]
    lca.dat$food.col = lca.dat[,food.col.lcadat]
    dat$percent.col = dat[,percent.col]
    
    
    dat <-
      left_join(dat %>% unique(.),
                lca.dat) %>%
      mutate(Bio_100g = Bio/10 * percent.col/100,
             GHGs_100g = GHG/10 * percent.col/100,
             Eut_100g = Eut/10 * percent.col/100,
             WatScar_100g = WatScar/10 * percent.col/100) %>%
      group_by(id, product_name, variable) %>%
      dplyr::summarise(tot.percent.col = mean(tot.percent.col),
                Bio_100g = sum(Bio_100g, na.rm = TRUE),
                GHGs_100g = sum(GHGs_100g, na.rm = TRUE),
                Eut_100g = sum(Eut_100g, na.rm = TRUE),
                WatScar_100g = sum(WatScar_100g, na.rm = TRUE))
    
   # updating values and returning data set
    dat[,tot.percent.col] = dat$tot.percent.col
    
    dat <-
      dat %>%
      dplyr::select(-tot.percent.col)
    
    dat <-
      dat %>%
      mutate(percent = ifelse(id_embedded %in% dat.count$id_embedded, 100, percent)) %>%
      mutate(tot_percent = ifelse(id_embedded %in% dat.count$id_embedded, 100, tot_percent))
    
    return(dat)
    
  }



# Matching products based on key search terms
match.products.function <-
  function(dat, search.products) {
    dat <-
      dat %>%
      mutate(Food_Category = NA)
    
    # Converting to character strings
    for(i in which(names(search.products) %in% '1') : which(names(search.products) %in% '500')) {
      search.products[,i] <- as.character(search.products[,i])
      convert.na = which(search.products[,i] == '')
      
      if(length(convert.na > 0)) {
        search.products[convert.na,i] <- NA
      }
    }
    
    # rex(search.words$Search_Words[11])
    cols.loop = which(names(search.products) %in% '1') : which(names(search.products) %in% '500')
    
    
    # Splitting into two groups
    # one data frame for words to search for
    # The second data frame for words to drop
    search.products.keep <-
      search.products %>%
      filter(!is.nan(Convert_to_na))
    
    search.products.drop <-
      search.products %>%
      filter(is.nan(Convert_to_na))
    
    for(i in 1:nrow(search.products.keep)) {
      # for(i in 1:5) {
      for(j in cols.loop) {
        if(!is.na(search.products.keep[i,j])) {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.products.keep[i,j], ignore.case = TRUE, product_name) ~ search.products.keep$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      search.products.keep$count[i] <- # Counting foods
        sum(!is.na(dat$Food_Category))
      
      dat$Food_Category = NA # Converting back to NAs
    }
    
    # Reordering based on how common words are
    search.products.keep <- search.products.keep[order(search.products.keep$count),]
    
    for(i in 1:nrow(search.products.keep)) {
      # for(i in 1:31) {
      for(j in cols.loop) {
        if(!is.na(search.products.keep[i,j])) {
          dat <-
            dat %>%
            mutate(Food_Category = 
                     ifelse(is.na(Food_Category),
                            case_when(grepl(search.products.keep[i,j], ignore.case = TRUE, product_name) ~ search.products.keep$LCA_Category[i]),
                            Food_Category))
          
        } # End of if statement
      } # End of loop through foods
      
      # And now removing products that should not be sorted into the category
      if(search.products.keep$LCA_Category[i] %in% search.products.drop$LCA_Category) {
        # Creating df with search words to drop
        tmp = search.products.drop %>% filter(LCA_Category %in% search.products.keep$LCA_Category[i]) %>% as.data.frame()
        for(j in cols.loop) {
          if(!is.na(tmp[1,j])) {
            # Getting indices to drop
            food.cat = which(dat$Food_Category %in% search.products.keep$LCA_Category[i])
            drop.indices = grep(tmp[1,j], dat$product_name, ignore.case = TRUE)
            # Getting intersection
            food.cat <- food.cat[food.cat%in%drop.indices]
            if(length(food.cat) > 0) {
              dat$Food_Category[food.cat] <- NA
            }
          }
        }
      }
      # search.products.keep$count[i] <- # Counting foods
      #   sum(!is.na(dat$Food_Category))
      # 
      # dat$Food_Category = NA # Converting back to NAs
    }
    
    return(dat)
  }




###
# Function to extract info where product lists e.g. 200g tomato per 100g product
set.gram.function <-
  function(dat,
           column.extract) {
    
    dat1 = as.data.frame(dat1)
    
    # Products that contain xxg per 100g product
    row.index <-
      c(grep("[0-9]{1,3}(\\s)?g(\\s)per(\\s)?[0-9]{1,3}", dat1[,column.extract], ignore.case = TRUE),
        grep("[0-9]{1,3}(\\s)?g(\\s)[^0-9]{1,}per(\\s)?[0-9]{1,3}", dat1[,column.extract], ignore.case = TRUE)) 
    row.index = sort(unique(row.index))
    
    # Creating data frame of these products with product name and ingredient text
    out.df <-
      dat1[row.index,c('id','product_name',column.extract)]
    
    # Everything between every comma
    tmp1 = str_extract_all(out.df[,column.extract],
                          ",.*,")
    # Getting location of last comma
    comma.extract = str_locate_all(out.df[,column.extract],",")
    
    # Looping to add to tmp1
    for(i in 1:nrow(out.df)) {
      last.comma = max(comma.extract[[i]][,1])
      tmp2 = substr(out.df[,column.extract][i], last.comma, nchar(out.df[,column.extract][i]))
      tmp1[[i]] <- append(tmp1[[i]],tmp2)
    }
    # Getting substr after last comma
    tmp2 = substr(out.df[,column.extract, comma.])
    
    # Getting all locations of xxg per 100g product
    # Getting this in two ways
    values.end.string1 =
      str_locate_all(dat1[row.index,column.extract],"[0-9]{1,3}(\\s)?g(\\s)per(\\s)?[0-9]{1,3}")
    
    values.end.string2 =
      str_locate_all(dat1[row.index,column.extract],"[0-9]{1,3}(\\s)?g(\\s)[^0-9]{1,}per(\\s)?[0-9]{1,3}")
    
    # Getting max list of values extracted
    for(i in 1:length(values.end.string)) {
      max.length = length(values.end.string[[i]])/2
    }
    
    # Adding columns to out.df
    out.df[,paste0('value_100g_',1:max.length)] = NA
    
    # Looping
    for (i in 1:nrow(out.df)) {
      # Tmp index from longer string
      tmp.index = values.end.string[[i]] %>% as.data.frame(.) %>% .[,1]
      tmp.string = out.df$ingredients_text[i]
      
      # Looping through
      for(j in (length(tmp.index)) : 1) {
        # Finding preceeding comma
        tmp.last.comma <- str_locate_all(tmp.string,",") %>% as.data.frame(.) %>% .[,1]
        
        # 
        tmp.last.comma.before <-
          tmp.last.comma < tmp.index[j]
        tmp.last.comma.before <-
          max(tmp.last.comma[(tmp.last.comma.before == TRUE)])
        #
        if(tmp.last.comma.before == max(tmp.last.comma)) {
          tmp.substr = substr(out.df$ingredients_text[i], 
                              start = tmp.last.comma.before+1, 
                              stop = nchar(out.df$ingredients_text[i]))
        } else {
          tmp.substr = substr(out.df$ingredients_text[i], 
                              start = tmp.last.comma.before+1, 
                              stop = tmp.last.comma[which(tmp.last.comma == tmp.last.comma.before) + 1])
        }
        
        # Adding to data frame
        out.df[i,j+3] <- tmp.substr
          
        
      }
      
      #
      
      
      
    }
    
    # Taking substring
    
    # Repeating (do this in a loop)
    
    # Identifying food category
    
    # Returning this list
    
    
    
    
    
  }




#####
###
# Function to interpolate % composition of food ingredients
interpolate.food.ingredients.validate <-
  function(dat.validate) {
    
    # Id for validating
    id.validate = 1

    # list of variables to loop through
    variable.list = paste0('V',1:length(unique(dat$variable)))
    
    # Some data management
    dat <- dat.long %>% mutate(percent = percent0)
    dat.validate <- dat.validate %>% mutate(percent = percent0)
    
    # Logical sense check for percentages
    for(i in variable.list) { # percent composition cannot be greater than 100/n, where n = order of ingredient in the product
      dat$percent[dat$variable == i & dat$percent > 100/which(variable.list == i) & !is.na(dat$percent)] <- NA
      dat.validate$percent[dat.validate$variable == i & dat.validate$percent > 100/which(variable.list == i) & !is.na(dat.validate$percent)] <- NA
    }
    
    # Creating new column with identified percent composition
    # Identifies rows where percent was listed in back of package information
    dat <-
      dat %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    # Saving this under a different name to avoid having to change naming conventions in the script below
    dat.keep <- dat
    
    dat.validate <-
      dat.validate %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    # Getting average composition by department, aisle, and shelf
    # Department
    department.sum <-
      dat.keep %>%
      group_by(Department, Food_Category, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>% filter(Department != 'NULL')
    
    # Aisle
    aisle.sum <-
      dat.keep %>%
      group_by(Aisle, Food_Category, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>% filter(Aisle != 'NULL')
    
    # Shelf
    shelf.sum <-
      dat.keep %>%
      group_by(Aisle, Shelf, Food_Category, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                count = dplyr::n()) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count > 10) %>% dplyr::select(-count) %>% filter(Shelf != 'NULL')
    
    
    # Creating data frame to save output data
    dat.out <- data.frame()
    
    # Looping through products with n ingredients
    for(n_ings in sort(unique(dat.validate$n_ingredients))) {
      
      # Data set with n ingredients in the product
      dat.validate.loop.n_ings <- 
        dat.validate %>% 
        filter(n_ingredients %in% n_ings)
      
      # Looping to randomly remove ingredients for which we know composition info
      for(max.loop in 2:n_ings) {
        # Getting all potential unique permutations of ingredients to drop
        perms = (combinations(n = n_ings, r = max.loop, v=paste0("V",1:n_ings), set=TRUE, repeats.allowed=FALSE)) %>% as.data.frame(.)
        
        # Getting max number of permutations
        max.perms = min(100, nrow(perms))
        if(max.perms == 100) {
          # dplyr::sample_n is throwing me an error
          # So I'm doing this old way
          # perms = dplyr::sample_n(tbl = perms, size = max.perms, replace = FALSE) %>% as.data.frame(.)
          perms = perms[sample(x = 1:nrow(perms), size = max.perms, replace = FALSE),] %>% as.data.frame(.)
        }
        
        # Looping through this max number of permutations
        for(perm in 1:max.perms) {
          # Dropping percent known from ingredients
          # And flipping "KEEP" for these ingredients to "NO"
          variables.drop = as.matrix(perms[perm,]) %>% as.vector(.)
          
          dat.validate.loop <-
            dat.validate.loop.n_ings %>%
            mutate(percent = ifelse(variable %in% variables.drop, NA, percent)) %>% # Dropping percent from randomized ingredients
            mutate(keep_percent = ifelse(is.na(percent), 'NO',keep_percent)) # Flipping "KEEP" for these ingredients to "NO" so that their composition can be updated
          
          # Updating tot_percent and n_ingredients for these items
          dat.percent.update <-
            dat.validate.loop %>%
            mutate(percent_check = ifelse(keep_percent %in% 'YES',1,0)) %>%
            group_by(id, product_name, Retailer, Department, Aisle, Shelf) %>%
            dplyr::summarise(percent_known = sum(percent, na.rm = TRUE),
                      n_percent_known = sum(percent_check))
          
          # Merging in these info
          # Need to save this to check how accuracy of method varies based on info known
          dat.validate.loop <-
            left_join(dat.validate.loop,
                      dat.percent.update)
          
          
          ###
          # And now starting the validation
          # Updating values based on shelf first, if more than 10 products in that shelf
          # But only if NAs
          # A lot of this is the same as or very similar to the interpolate.food.ingredients function above
          # But keeping the hwole thing here because of a few differences that make calling the function above impossible (without rewriting the whole thing)
          
          # Updating values based on the shelf, if more than 10 products in the shelf
          dat.validate.loop <-
            left_join(dat.validate.loop, shelf.sum %>% filter(!is.na(Food_Category))) %>%
            mutate(percent = ifelse(is.na(percent), percent_shelf, percent))
          
          # Updating values based on aisle, if more than 10 products in that aisle
          dat.validate.loop <-
            left_join(dat.validate.loop, aisle.sum %>% filter(!is.na(Food_Category))) %>%
            mutate(percent = ifelse(is.na(percent), percent_aisle, percent)) %>%
            dplyr::select(-percent_aisle)
          
          # Updating values based on department
          dat.validate.loop <-
            left_join(dat.validate.loop, department.sum %>% filter(!is.na(Food_Category))) %>%
            mutate(percent = ifelse(is.na(percent), percent_department, percent)) %>%
            dplyr::select(-percent_department)
          
          # Now updating values based on what is known
          # Getting list of products to interpolate (i.e. any product where total percent != 100) and n ingredients > 1
          product.percent <-
            dat.validate.loop %>%
            group_by(id.new, product_name) %>%
            dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
          
          # merging in
          # And setting order of ingredients in product
          dat.validate.loop <- 
            left_join(dat.validate.loop, product.percent) %>%
            transform(variable = factor(variable, levels = variable.list)) %>%
            mutate(tot_percent = tot_percent_new) %>%
            dplyr::select(-tot_percent_new)
          
          # Ordering
          dat.validate.loop <-
            dat.validate.loop[order(dat.validate.loop$id.new,dat.validate.loop$variable),]
          
          # And list of products to loop through
          product.list <- 
            product.percent$id.new[product.percent$tot_percent_new != 100]
          
          # Changing name of data frame so that I don't have to change naming conventions in the rest of the function
          dat <- dat.validate.loop
          
          # Adding info for the validation
          # Info for validation
          dat$info_validate <-
            paste0("Tot_Num_Ings: ", n_ings,
                   "; N_Ings_Removed: ", max.loop,
                   "; Perm_Removed: ", perm)
          
          # List of variables removed when validating
          dat$variable_removed_validate = str_c(variables.drop, collapse = ';')
          
          # Unique id
          dat$id_validate <- NA

          
          # Looping through these products to update percent composition
          # This is a behemoth of a loop
          for(j in product.list) {
            # Temporary data frame
            tmp.dat <- dat[dat$id.new %in% j,]
            
            # Easy exception
            # If only one "NO" in keep_percent, then that needs to change such that total percent sums to 100
            if(sum(tmp.dat$keep_percent == 'NO') == 1) {
              tmp.dat$percent[tmp.dat$keep_percent == 'NO'] <-
                100 - sum(tmp.dat$percent[tmp.dat$keep_percent == 'YES'])
              tmp.vector1 <- tmp.dat$percent
              tmp.keep <- tmp.dat$keep_percent
              
            } else if(sum(is.na(tmp.dat$percent)) == nrow(tmp.dat))  {
              tmp.dat$percent = 100/nrow(tmp.dat)
              tmp.vector1 <- tmp.dat$percent
              tmp.keep <- tmp.dat$keep_percent
            } else if (sum(tmp.dat$percent, na.rm = TRUE) == 0) {
              tmp.dat$percent = 100 / nrow(tmp.dat)
              tmp.vector1 <- tmp.dat$percent
              tmp.keep <- tmp.dat$keep_percent
            } else if (nrow(tmp.dat) == 1) {
              tmp.dat$percent = 100
              tmp.vector1 <- tmp.dat$percent
              tmp.keep <- tmp.dat$keep_percent
            } else {
              # Logic here is to interpolate other values
              # Getting vector of known percents
              tmp.vector <- tmp.dat$percent
              tmp.keep <- tmp.dat$keep_percent
              
              # Get total percent composition of tmp vector
              tot.percent <- sum(tmp.vector, na.rm = TRUE)
              
              ###
              # Interpolate normally
              # If sum tmp vector = 0, then cannot do anything
              # Therefore, if sum does not equal 0
              if(sum(tmp.vector, na.rm = TRUE) != 0) {
                tmp.vector1 = tmp.vector
                
                # Logic checks to fix percent composition of ingredients
                # (1) For example, if N2 == N4, then N3 = N4. Flip tmp.keep to "KEEP"
                # Doing this for every instance this appears
                if(length(tmp.vector1[tmp.keep %in% 'YES']) != length(unique(tmp.vector1[tmp.keep %in% 'YES']))) {
                  # Finding location of known composition that are equal
                  tmp.table <- 
                    table(tmp.vector1[tmp.keep %in% 'YES']) %>%
                    .[. != 1] %>% names(.) %>% as.numeric(.)
                  
                  for(x in 1:length(tmp.table)) {
                    tmp.indices <- which(tmp.vector1 == tmp.table[x] & tmp.keep %in% 'YES')
                    if(length(tmp.indices) > 0) {
                      tmp.vector1[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <-
                        tmp.vector1[tmp.indices[1]]
                      tmp.keep[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <- "YES"
                    }
                  }
                  
                }
                
                
                # Getting indices with reported percents
                tmp.check <- which(tmp.vector != 0)
                
                # But can only interpolate if more than 1 ingredient is recognized with a percent
                # if(length(tmp.check) > 1) {
                if(length(tmp.check) > 1) {
                  for(k in 1:(length(tmp.check) - 1)) {
                    if(!(k %in% c(0))) {
                      tmp.vector1[tmp.check[k]:tmp.check[k+1]] <-
                        seq(from = tmp.vector1[tmp.check[k]], to = tmp.vector1[tmp.check[k+1]], length.out = (tmp.check[k+1]-tmp.check[k] + 1)) 
                    }
                  }
                }
                
                
                # # Updating ingredients for first listed ingredient
                # if(min(tmp.check) != 1) {
                #   tmp.vector1[1:(min(tmp.check)-1)] <-
                #     tmp.vector1[min(tmp.check)]
                # }
              }
              
              ###
              # And exception if total sum of all ingredients > 100 when interpolating normally
              # Only doing this for the last gap in the ingredients list
              # E.g. if ingredients 5-9 are unknown, interpolate these normally
              # But updated interpolation for ingredients 1-3
              # Doing it this way because there is normally a big dropoff in percent composition after the first ingredient or two
              
              
              
              # if(sum(tmp.vector1) > 100) { # If true, then total composition of all ingredients recognized is greater than 100
              #   # Only updating the first gap in the ingredient list
              #   # Assuming that the other graps have been properly managed
              #   
              #   # Removing values that were identified
              #   tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 0
              #   
              #   # Getting sum composition of the other products
              #   # Remaining ingredients need to make this sum to 100
              #   tmp.sum <- sum(tmp.vector1)
              #   
              #   # If only 1 missing ingredient, then all of the missing values are assigned to this
              #   if(length((tmp.check[1] + 1) : (tmp.check[2] -1)) == 1) {
              #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 100 - tmp.sum
              #   } else { # if more than 1 missing ingredient
              #     # Getting what the interpolation would be
              #     tmp.int <- seq(from = tmp.vector1[tmp.check[1]], to = tmp.vector1[tmp.check[2]], length.out = (tmp.check[2] - tmp.check[1]) + 1)
              #     # Getting only the 2nd to nth values
              #     tmp.int <- tmp.int[2:(length(tmp.int)-1)]
              #     # Now weighting missing ingredients by the values they would be if interpolation was correct
              #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- # Index of missing values
              #       (100-tmp.sum) * # Total missing percent
              #       tmp.int / # This is what the interpolation would be if the total ingredient list didn't sum to 100
              #       sum(tmp.int) # Then dividing by sum of this interpolation to get percent of missing ingredients
              #   }
              # }
              
              
              ###
              # Updating percentages of last n ingredients after the last ingredient that has a percent recognized
              if(max(tmp.check) < nrow(tmp.dat)) {
                
                tmp.keep[length(tmp.keep)] <- 'YES'
                # Getting how these would interpolate, assuming the 10th ingredient is .1%
                tmp.int <- seq(from = tmp.vector1[tmp.check[length(tmp.check)]], # Value of the last recognized ingredient
                               to = min(c(.1,tmp.vector1[tmp.check[length(tmp.check)]])), # Assuming that the last ingredient is the lesser of 1% or last identified % composition
                               length.out = (nrow(tmp.dat)-tmp.check[length(tmp.check)] + 1)) # Number of ingredients needed
                
                if(length(tmp.int) >= 1) { # Logic check in case there aren't any more ingredients to identify
                  # Updating tmp.vector
                  tmp.vector1[(tmp.check[length(tmp.check)] + 1) : (tmp.check[length(tmp.check)] + length(tmp.int) - 1)] <-
                    tmp.int[2:length(tmp.int)]
                }
              }
              
              
              
              ###
              # Updating percentages or first n ingredients if only one percentage is recognized
              if(min(tmp.check) != 1) { # Logic statement to identify products where the first identified ingredient is not the first
                tmp.sum <- sum(tmp.vector1, na.rm = TRUE) # total composition of ingredients
                tmp.int <- seq(from = (100-tmp.sum), # Percent not yet recognized
                               to = tmp.vector1[min(tmp.check)], # Percent composition of the first product
                               length.out = (min(tmp.check)))
                if(length(tmp.int)>=1) {
                  tmp.vector1[1:(min(tmp.check))] <- tmp.int
                }
              }
              
              ###
              # Looping through to correct values where i.e. v1 < v2, etc
              # But only adjusting the non-keep percentages
              for(z in (length(tmp.vector1) - 1):1) {
                if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
                  tmp.vector1[z] <- tmp.vector1[z+1]
                }
              }
              
              ###
              # Assuming the last ingredient is correct
              # Need to do this to provide a counterpoint for everything else
              # And also serves as a basis to solve the rest of this puzzle
              tmp.keep[length(tmp.keep)] <- 'YES'
              
              ###
              # And now correcting percent composition when they do not sum to 100
              # Doing this in a series of while loops
              # Where logic is
              # (a) Adjust the "non-keep" ingredients such that everything sums to 100
              # (b) check that e.g. v1 >= v2, etc
              # (c) check that e.g. v2 >= 100/2
              counter = 1
              while(sum(tmp.vector1) != 100 & counter <= 10) {
                counter = counter + 1
                if(round(sum(tmp.vector1)) != 100) {
                  # Getting sum percent of the interpolated ingredients
                  tmp.sum.int <- sum(tmp.vector1[tmp.keep == 'NO'])
                  tmp.sum <- sum(tmp.vector1[tmp.keep == 'YES'])
                  # And scaling so that all ingredients sum to 100
                  tmp.vector1[tmp.keep == 'NO'] <-
                    tmp.vector1[tmp.keep == 'NO'] *
                    ((100 - tmp.sum) / tmp.sum.int)
                  
                  tmp.vector1[is.nan(tmp.vector1)] <- 0
                } # End if statement
                
                ###
                # Doing two few checks, specifically 
                # (1) %1 > %2 > %3 etc
                # Looping through to correct values where i.e. v1 < v2, etc
                # But only adjusting the non-keep percentages
                for(z in (length(tmp.vector1) - 1):1) {
                  if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
                    tmp.vector1[z] <- tmp.vector1[z+1]
                  }
                }
                
                
                # (2)
                # Updating keep percent when possible
                for(z in (length(tmp.vector1) - 1) : 1) {
                  if(tmp.vector1[z] == tmp.vector1[z+1] &
                     tmp.keep[z+1] == 'YES') {
                    tmp.keep[z] <- 'YES'
                  }
                }
                
                # (3) %n msut be less than 100/n
                # But only adjusting the non-identified ingredients
                # And then flagging ones that cannot change again
                for(i in length(tmp.vector1):2) {
                  if(tmp.vector1[i] > 100/i & tmp.keep == 'NO') {
                    tmp.vector1[i] <- 100/i
                  } # End for loop for ingredients
                }
                
              } # End while loop
            } # End of big if else loop
            
            
            
            # updating values in the data frame
            dat$percent[dat$id.new %in% j] <-
              tmp.vector1
            dat$tot_percent[dat$id.new %in% j] <- sum(tmp.vector1, na.rm = TRUE)
            
            # Updating validate id
            dat$id_validate[dat$id.new %in% j] <- id.validate
            # Adding 1 to counter
            id.validate = id.validate + 1
          } # End of loop for products
          
          # Stacking data used to validate estimates
          dat.out <- rbind(dat.out, dat)
        } # End of loop that cycles through which ingredients info is dropped
      } # End of loop that cycles through number of ingredients to drop
    } # End of loop that cycles through products with n ingredients
    
    
    # Dropping id.new
    # dat <- dat %>% dplyr::select(-id.new)
    # Returning data frame
    return(dat.out)
  } # End function for ingredient interpolating



# Getting all possible permutations of ingredients in each product
# Then dropping composition information for these products
# Doing this for validation purposes
validate.combos.function <- 
  function(dat.validate) {
    # Creating data frame to save output data
    dat.out <- data.frame()
    for(n_ings in sort(unique(dat.validate$n_ingredients))) {
      
      # updating column name to integrate with rest of code
      dat.validate <- dat.validate %>% mutate(percent = percent0)
      
      
      # counter for validation id
      id.validate = 1
      
      # Data set with n ingredients in the product
      dat.validate.loop.n_ings <- 
        dat.validate %>% 
        filter(n_ingredients %in% n_ings)
      
      # Looping to randomly remove ingredients for which we know composition info
      for(max.loop in 2:n_ings) {
        # Getting all potential unique permutations of ingredients to drop
        perms = (combinations(n = n_ings, r = max.loop, v=paste0("V",1:n_ings), set=TRUE, repeats.allowed=FALSE)) %>% as.data.frame(.)
        
        # Getting max number of permutations
        max.perms = min(100, nrow(perms))
        if(max.perms == 100) {
          # dplyr::sample_n is throwing me an error
          # So I'm doing this old way
          # perms = dplyr::sample_n(tbl = perms, size = max.perms, replace = FALSE) %>% as.data.frame(.)
          perms = perms[sample(x = 1:nrow(perms), size = max.perms, replace = FALSE),] %>% as.data.frame(.)
        }
        
        # Looping through this max number of permutations
        for(perm in 1:max.perms) {
          # Dropping percent known from ingredients
          # And flipping "KEEP" for these ingredients to "NO"
          variables.drop = as.matrix(perms[perm,]) %>% as.vector(.)
          
          dat.validate.loop <-
            dat.validate.loop.n_ings %>%
            mutate(percent = ifelse(variable %in% variables.drop, NA, percent)) %>% # Dropping percent from randomized ingredients
            mutate(keep_percent = ifelse(is.na(percent), 'NO','YES')) # Flipping "KEEP" for these ingredients to "NO" so that their composition can be updated
          
          # Updating tot_percent and n_ingredients for these items
          dat.percent.update <-
            dat.validate.loop %>%
            mutate(percent_check = ifelse(keep_percent %in% 'YES',1,0)) %>%
            group_by(id, product_name, Retailer, Department, Aisle, Shelf) %>%
            dplyr::summarise(percent_known = sum(percent, na.rm = TRUE),
                             n_percent_known = sum(percent_check))
          
          # Merging in these info
          # Need to save this to check how accuracy of method varies based on info known
          dat.validate.loop <-
            left_join(dat.validate.loop,
                      dat.percent.update) %>%
            mutate(variable_removed_validate = paste0(variables.drop, collapse = ';')) %>%
            mutate(info_validate = paste0('Tot_Num_Ings: ',n_ings,
                                          '; N_Ings_Removed: ',max.loop,
                                          '; Perm_Removed: ',perm)) %>%
            mutate(id_validate = id.validate)
          # Adding one to validated id
          id.validate = id.validate + 1
          
          dat.out <- rbind(dat.out, dat.validate.loop)
        }
      }
    }
    
    # And rbinding actual data
    # used to update estimates when interpolating percent composition
    dat.out <-
      rbind(dat.out, 
            dat.validate %>% mutate(percent_known = 100, n_percent_known = n_ingredients, variable_removed_validate = 'none',info_validate = 'none', id_validate = 'none'))
    
    return(dat.out)
  }




# Used for estimating composition of ingredients
# Both when validating and when first assessing
# Note that this function also calls on dat.keep, which is not used as a specific input into the function
# Dat.keep is created in the primary wrapper
# This is done so that it can be parallelised with mclapply if needed
interpolate.food.ingredients.trial <-
  function(dat) {
    
    # list of variables to loop through
    variable.list = paste0('V',1:length(unique(dat$variable)))
    
    # Some data management
    dat <- 
      dat %>% 
      mutate(percent = percent0) %>%
      mutate(Aisle_Shelf = paste0(Aisle, "_", Shelf)) %>%
      mutate(percent_estimated = NA)
    
    # Saving this to avoid updating averages in the department/aisle/shelf based on already updated averages
    dat.keep <- dat.percent.save %>% mutate(Aisle_Shelf = paste0(Aisle,"_",Shelf)) %>% filter(!is.na(Food_Category))
    
    # Logical sense check for percentages
    for(i in variable.list) { # percent composition cannot be greater than 100/n, where n = order of ingredient in the product
      dat$percent[dat$variable == i & dat$percent > 100/which(variable.list == i) & !is.na(dat$percent)] <- NA
    }
    
    # Creating new column with identified percent composition
    # Identifies rows where percent was listed in back of package information
    dat <-
      dat %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    ### Updating salt info
    # Check is:
    #  if food_category recognised as salt
    # and if estimated composition of salt > food packaging, 
    # then update salt content info
    # and set keep to 'YES'
    # else, set keep to 'YES
    dat <-
      dat %>%
      left_join(.,salt.dat %>% dplyr::select(id = product_id, salt_amount_g)) %>% # merging in salt dat
      mutate(salt_amount_g = ifelse(Food_Category %in% 'Salt', salt_amount_g, NA)) %>% # converting to NA if the ingredient isn't actually salt...
      mutate(percent = ifelse(is.na(percent) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%
      mutate(percent = ifelse(percent > salt_amount_g & !is.na(salt_amount_g) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%# updating salt percent
      mutate(keep_percent = ifelse(Food_Category %in% 'Salt' & !is.na(dat$percent), 'YES',keep_percent)) # and setting keep_percent for salt to 'YES
    
    # And doing checks on number of ingredients in each food category, sub category, and sub sub category
    # The general rule is that if there are >= 10 observations, then use info from the sub sub category, if not, use info from the sub cateogry, and if not, use info from the original category
    # Getting this aggregation by shelf, aisle, and department
    
    # By department
    count.dep <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Food_Category) %>%
                  dplyr::summarise(count_dep_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_dep_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_dep_sub_sub = n())) %>%
      filter(!is.na(Food_Category))  %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_dep_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_dep_sub)) %>%
      mutate(count_dep_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_dep_sub_sub))
    # By aisle
    count.ais <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Aisle, Food_Category) %>%
                  dplyr::summarise(count_ais_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Aisle, Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_ais_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Aisle, Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_ais_sub_sub = n())) %>%
      filter(!is.na(Food_Category)) %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_ais_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_ais_sub)) %>%
      mutate(count_ais_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_ais_sub_sub))
    
    # By shelf
    count.shelf <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category) %>%
                  dplyr::summarise(count_shelf_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_shelf_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_shelf_sub_sub = n())) %>%
      filter(!is.na(Food_Category)) %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_shelf_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_shelf_sub)) %>%
      mutate(count_shelf_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_shelf_sub_sub))
    
    
    # Estimating percent composition using info from similar products and the same food category ----
    # Updating values based on shelf first, if more than 10 products in that shelf
    # But only if NAs
    shelf.sum <-
      dat.keep %>%
      left_join(.,count.shelf) %>%
      mutate(Food_Category_sub = 
               ifelse(count_shelf_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_shelf_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department, Aisle, Shelf,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Shelf, Food_Category, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shelf.sum %>% filter(!is.na(Food_Category)))
    
    # Updating values based on aisle, if more than 10 products in that aisle
    aisle.sum <-
      dat.keep %>%
      left_join(.,count.ais) %>%
      mutate(Food_Category_sub = 
               ifelse(count_ais_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_ais_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department, Aisle,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Food_Category, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable)), NA, percent_aisle))
    
    dat <-
      left_join(dat, aisle.sum %>% filter(!is.na(Food_Category)))
    
    # Updating values based on department
    department.sum <-
      dat.keep %>%
      left_join(.,count.dep) %>%
      mutate(Food_Category_sub = 
               ifelse(count_dep_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_dep_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, Food_Category, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department))
    
    dat <-
      left_join(dat, department.sum %>% filter(!is.na(Food_Category)))
    
    # Repeating the above across all food categories ----
    shel.sum <-
      dat.keep %>%
      group_by(Aisle, Shelf, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shel.sum %>% dplyr::rename(percent_shelf_all = percent_shelf))
    
    
    ais.sum <-
      dat.keep %>%
      group_by(Aisle, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable)), NA, percent_aisle))
    
    dat <-
      left_join(dat, ais.sum %>% dplyr::rename(percent_aisle_all = percent_aisle))
    
    dep.sum <- 
      dat.keep %>%
      group_by(Department, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department))
    
    dat <-
      left_join(dat, dep.sum %>% dplyr::rename(percent_department_all = percent_department))
    
    # Now interpolating based on power law regression equations ----
    # Doing this by building regression models for each Aisle and Shelf
    # Structure of model is percent = 1/variable^2, w/ no intercept to avoid negative values
    # Where variable is the location of variable in the product's ingredient list
    # Starting with each shelf
    # List to loop
    loop.list <- unique(dat.keep$Aisle_Shelf)
    dat$percent_estimated_shelf = NA
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        left_join(.,count.shelf) %>%
        mutate(Food_Category_sub = 
                 ifelse(count_shelf_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
        mutate(Food_Category = 
                 ifelse(count_shelf_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
        dplyr::select(id, product_name, Retailer, Department, Aisle, Shelf,
                      variable, variablen, value, Food_Category, percent, Aisle_Shelf) %>%
        filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
        filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
        unique(.) %>%
        filter(Aisle_Shelf %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z & dat.keep$variablen <= 15]) 
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z]) >= 5) {
        # tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        
        lm.list <-
          list(lm(tmp.dat$percent ~ I(1/tmp.dat$variablen^2) - 1),
               lm(tmp.dat$percent ~ I(1/tmp.dat$variablen) - 1),
               lm(tmp.dat$percent ~ tmp.dat$variablen))# No intercept
        r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
        # p.list <- c(summary(lm.list[[1]])$coefficients[4],summary(lm.list[[2]])$coefficients[4])
        
        tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
        # Only updating if adj R^2 > .75 and P < .05
        # Second check to get rid of nan statements
	if(!is.nan(summary(tmp.lm)$adj.r.squared)) {
	if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
          if(which(r2.list %in% max(r2.list)) %in% 1) {
            dat <-
              dat %>%
              mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen^2,
                                                      percent_estimated))
          } else if (which(r2.list %in% max(r2.list)) %in% 2) {
            dat <-
              dat %>%
              mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen,
                                                      percent_estimated))
          } else {
            dat <-
              dat %>%
              mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                      tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen,
                                                      percent_estimated))
          }
          
	}
        }
      }
    }
    
    # Repeating for Aisles
    # Setting percent_estimated back to 0
    dat <-
      dat %>%
      mutate(percent_estimated_aisle = NA)
    # List to loop
    loop.list <- unique(dat.keep$Aisle)
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        left_join(.,count.ais) %>%
        mutate(Food_Category_sub = 
                 ifelse(count_ais_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
        mutate(Food_Category = 
                 ifelse(count_ais_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
        dplyr::select(id, product_name, Retailer, Department, Aisle,
                      variable, variablen, value, Food_Category, percent) %>%
        filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
        filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
        unique(.) %>%
        filter(Aisle %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle %in% z & dat.keep$variablen <= 15])
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle %in% z]) >= 5) {
        # tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        
        lm.list <-
          list(lm(tmp.dat$percent ~ I(1/tmp.dat$variablen^2) - 1),
               lm(tmp.dat$percent ~ I(1/tmp.dat$variablen) - 1),
               lm(tmp.dat$percent ~ tmp.dat$variablen))# No intercept
        r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
        tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
        # Only updating if adj R^2 > .5 and P < .05
        if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
          if(which(r2.list %in% max(r2.list)) %in% 1) {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen^2,
                                                      percent_estimated))
          } else if (which(r2.list %in% max(r2.list)) %in% 2) {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen,
                                                      percent_estimated))
          } else {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen,
                                                      percent_estimated))
          }
          
          
        }
      }
    }
    
    # Repeating for Departments
    # Don't need a loop, because this already loops across departments
    
    
    # Repeating for departments
    dep.sum <- 
      dat.keep %>%
      left_join(.,count.dep) %>%
      mutate(Food_Category_sub = 
               ifelse(count_dep_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_dep_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department,
                    variable, variablen, value, Food_Category, percent) %>%
      filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
      filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, variable, variablen) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      mutate(percent_department = ifelse(count < 10, NA, percent_department)) %>% 
      dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department)) %>%
      mutate(variablen = as.numeric(gsub("V","",variable)))
    
    prop_variables = length(unique(dep.sum$variablen[!is.na(dep.sum$percent_department)])) / max(dat.keep$variablen[dat.keep$variablen <= 15])
    
    dat$percent_estimated_department = NA
    
    # Building regression to estimate composition of other ingredients
    if(prop_variables >= 0.5 & max(dep.sum$variablen) >= 5) {
      lm.list <-
        list(lm(dat.keep$percent ~ I(1/dat.keep$variablen^2) - 1),
             lm(dat.keep$percent ~ I(1/dat.keep$variablen) - 1),
             lm(dat.keep$percent ~ dat.keep$variablen))# No intercept
      r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
      # p.list <- c(summary(lm.list[[1]])$coefficients[4],summary(lm.list[[2]])$coefficients[4])
      
      tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
      
      # tmp.lm = lm(dat.keep$percent ~ I(1/dat.keep$variablen) - 1) # No intercept
      # Predicting
      if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
        # dep.sum <-
        #   dep.sum %>%
        #   mutate(percent_estimated = tmp.lm$coefficients[[1]] / variablen^2) %>% # Predicting 
        #   mutate(percent_department = ifelse(is.na(percent_department), percent_estimated, percent_department)) %>%
        #   dplyr::select(variable, percent_department)
        # Merging back in
        if(which(r2.list %in% max(r2.list)) %in% 1) {
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] / variablen^2) 
        } else if (which(r2.list %in% max(r2.list)) %in% 2 ){
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] / variablen)
        } else {
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen) %>%
            mutate(check = predict(tmp.lm,dat))
        }
        
      }
    }
    
    # Calculating absolute difference to use for a given food category and ingredient number
    dat <-
      dat %>%
      mutate(abs_difference = NA) %>%
      mutate(shelf_dif = abs(log2(percent / percent_shelf)),
             aisle_dif = abs(log2(percent / percent_aisle)),
             dep_dif = abs(log2(percent / percent_department)),
             shelf_dif_all = abs(log2(percent / percent_shelf_all)),
             aisle_dif_all = abs(log2(percent / percent_aisle_all)),
             dep_dif_all = abs(log2(percent / percent_aisle_all)),
             shelf_dif_estimated = abs(log2(percent / percent_estimated_shelf)),
             aisle_dif_estimated = abs(log2(percent / percent_estimated_aisle)),
             dep_dif_estimated = abs(log2(percent / percent_estimated_department)))
    
    ### Now getting best average fit for:
    # And assigning which approach to use based on the below ordering
    # E.g. if data on (1) is available, use that, if not, use data from (2), if not, then data from (3), etc
    # With additional quality checks - e.g. log2 ratio < .2
    # (1) food category by ingredient number by aisle and shelf
    # (2) ingredient number by aisle and shelf
    # (3) food category by ingredient number by aisle
    # (4) ingredient number by aisle
    # (5) food category by ingredient number by department
    # (6) ingredient number by department
    # (7) shelf
    # (8) aisle
    # (9) department
    
    ### Shelf by ingredient
    food.ingredient.fit.shelf.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen, Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_dif = mean(shelf_dif,na.rm=TRUE),
                       aisle_dif = mean(aisle_dif,na.rm=TRUE),
                       dep_dif = mean(dep_dif,na.rm=TRUE),
                       shelf_dif_all = mean(shelf_dif_all,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf.variable.food[ii,c(which(names(food.ingredient.fit.shelf.variable.food) %in% 'shelf_dif'):which(names(food.ingredient.fit.shelf.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf.variable.food[ii,c(which(names(food.ingredient.fit.shelf.variable.food) %in% 'shelf_dif'):which(names(food.ingredient.fit.shelf.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf.variable.food$which.min.value.shelf.food.variable = which.min.value
    food.ingredient.fit.shelf.variable.food$which.min.out.shelf.food.variable = names(which.min.out)
    
    # And getting rid of NAs
    food.ingredient.fit.shelf.variable.food <-
      food.ingredient.fit.shelf.variable.food %>%
      mutate(which.min.out.shelf.food.variable = ifelse(which.min.value.shelf.food.variable %in% 99999, NA, which.min.out.shelf.food.variable),
             which.min.value.shelf.food.variable = ifelse(which.min.value.shelf.food.variable %in% 99999, NA, which.min.value.shelf.food.variable),)
    
    ### Shelf by ingredient number
    food.ingredient.fit.shelf.variable <- 
      dat %>% 
      group_by(variablen, Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_dif_all = mean(shelf_dif_all,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf.variable[ii,c(which(names(food.ingredient.fit.shelf.variable) %in% 'shelf_dif_all'):which(names(food.ingredient.fit.shelf.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf.variable[ii,c(which(names(food.ingredient.fit.shelf.variable) %in% 'shelf_dif_all'):which(names(food.ingredient.fit.shelf.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf.variable$which.min.value.shelf.variable = which.min.value
    food.ingredient.fit.shelf.variable$which.min.out.shelf.variable = names(which.min.out)
    
    food.ingredient.fit.shelf.variable <-
      food.ingredient.fit.shelf.variable %>%
      mutate(which.min.out.shelf.variable = ifelse(which.min.value.shelf.variable %in% 99999, NA, which.min.out.shelf.variable),
             which.min.value.shelf.variable = ifelse(which.min.value.shelf.variable %in% 99999, NA, which.min.value.shelf.variable),)
    
    ### Aisle by ingredient and food
    food.ingredient.fit.aisle.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen, Aisle) %>% # Getting averages
      dplyr::summarise(aisle_dif = mean(aisle_dif,na.rm=TRUE),
                       dep_dif = mean(dep_dif,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle.variable.food[ii,c(which(names(food.ingredient.fit.aisle.variable.food) %in% 'aisle_dif'):which(names(food.ingredient.fit.aisle.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle.variable.food[ii,c(which(names(food.ingredient.fit.aisle.variable.food) %in% 'aisle_dif'):which(names(food.ingredient.fit.aisle.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle.variable.food$which.min.value.aisle.food.variable = which.min.value
    food.ingredient.fit.aisle.variable.food$which.min.out.aisle.food.variable = names(which.min.out)
    # And updating NAs
    food.ingredient.fit.aisle.variable.food <-
      food.ingredient.fit.aisle.variable.food %>%
      mutate(which.min.out.aisle.food.variable = ifelse(which.min.value.aisle.food.variable %in% 99999, NA, which.min.out.aisle.food.variable),
             which.min.value.aisle.food.variable = ifelse(which.min.value.aisle.food.variable %in% 99999, NA, which.min.value.aisle.food.variable),)
    
    ### Aisle by ingredient number
    food.ingredient.fit.aisle.variable <- 
      dat %>% 
      group_by(variablen, Aisle) %>% # Getting averages
      dplyr::summarise(aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle.variable[ii,c(which(names(food.ingredient.fit.aisle.variable) %in% 'aisle_dif_all'):which(names(food.ingredient.fit.aisle.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle.variable[ii,c(which(names(food.ingredient.fit.aisle.variable) %in% 'aisle_dif_all'):which(names(food.ingredient.fit.aisle.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle.variable$which.min.value.aisle.variable = which.min.value
    food.ingredient.fit.aisle.variable$which.min.out.aisle.variable = names(which.min.out)
    # And updating NAs
    food.ingredient.fit.aisle.variable <-
      food.ingredient.fit.aisle.variable %>%
      mutate(which.min.out.aisle.variable = ifelse(which.min.value.aisle.variable %in% 99999, NA, which.min.out.aisle.variable),
             which.min.value.aisle.variable = ifelse(which.min.value.aisle.variable %in% 99999, NA, which.min.value.aisle.variable),)
    
    ### Department by ingredient and food
    food.ingredient.fit.dep.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen) %>% # Getting averages
      dplyr::summarise(dep_dif = mean(dep_dif, na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.dep.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.dep.variable.food[ii,c(which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_dif'):which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.dep.variable.food[ii,c(which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_dif'):which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.dep.variable.food$which.min.value.dep.variable.food = which.min.value
    food.ingredient.fit.dep.variable.food$which.min.out.dep.variable.food = names(which.min.out)
    
    # And updating NAs
    food.ingredient.fit.dep.variable.food <-
      food.ingredient.fit.dep.variable.food %>%
      mutate(which.min.out.dep.variable.food = ifelse(which.min.value.dep.variable.food %in% 99999, NA, which.min.out.dep.variable.food),
             which.min.value.dep.variable.food = ifelse(which.min.value.dep.variable.food %in% 99999, NA, which.min.value.dep.variable.food),)
    
    ### Department by ingredient number
    food.ingredient.fit.dep.variable <- 
      dat %>% 
      group_by(variablen) %>% # Getting averages
      dplyr::summarise(dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.dep.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.dep.variable[ii,c(which(names(food.ingredient.fit.dep.variable) %in% 'dep_dif_all'):which(names(food.ingredient.fit.dep.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.dep.variable[ii,c(which(names(food.ingredient.fit.dep.variable) %in% 'dep_dif_all'):which(names(food.ingredient.fit.dep.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.dep.variable$which.min.value.dep.variable = which.min.value
    food.ingredient.fit.dep.variable$which.min.out.dep.variable = names(which.min.out)
    # Updating NAs
    food.ingredient.fit.dep.variable <-
      food.ingredient.fit.dep.variable %>%
      mutate(which.min.out.dep.variable = ifelse(which.min.value.dep.variable %in% 99999, NA, which.min.out.dep.variable),
             which.min.value.dep.variable = ifelse(which.min.value.dep.variable %in% 99999, NA, which.min.value.dep.variable),)
    
    ### By Shelf
    food.ingredient.fit.shelf <- 
      dat %>% 
      group_by(Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf[ii,c(which(names(food.ingredient.fit.shelf) %in% 'shelf_est'):which(names(food.ingredient.fit.shelf) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf[ii,c(which(names(food.ingredient.fit.shelf) %in% 'shelf_est'):which(names(food.ingredient.fit.shelf) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf$which.min.value.shelf = which.min.value
    food.ingredient.fit.shelf$which.min.out.shelf = names(which.min.out)
    # Updating NAs
    food.ingredient.fit.shelf <-
      food.ingredient.fit.shelf %>%
      mutate(which.min.out.shelf = ifelse(which.min.value.shelf %in% 99999, NA, which.min.out.shelf),
             which.min.value.shelf = ifelse(which.min.value.shelf %in% 99999, NA, which.min.value.shelf),)
    ### By Aisle
    food.ingredient.fit.aisle <- 
      dat %>% 
      group_by(Aisle) %>% # Getting averages
      dplyr::summarise(aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle[ii,c(which(names(food.ingredient.fit.aisle) %in% 'aisle_est'):which(names(food.ingredient.fit.aisle) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle[ii,c(which(names(food.ingredient.fit.aisle) %in% 'aisle_est'):which(names(food.ingredient.fit.aisle) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle$which.min.value.aisle = which.min.value
    food.ingredient.fit.aisle$which.min.out.aisle = names(which.min.out)
    # Removing NAs
    food.ingredient.fit.aisle <-
      food.ingredient.fit.aisle %>%
      mutate(which.min.out.aisle = ifelse(which.min.value.aisle %in% 99999, NA, which.min.out.aisle),
             which.min.value.aisle = ifelse(which.min.value.aisle %in% 99999, NA, which.min.value.aisle),)
    
    ### By Department
    food.ingredient.fit.dep <- 
      dat %>% 
      group_by(Department) %>% # Getting averages
      dplyr::summarise(dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(which.min.value.dep = 'dep_est',
             which.min.out.dep = dep_est)
    
    ### Merging
    # To decide which approach to use
    # Go down order as described above
    # Whilst throwing in some quality checks (first checking if log ratio 2^.1, then 2^.25, and then 2^.5)
    # If none of the above, then estimating composition of that ingredient by interpolating
    dat <-
      dat %>% # Merging
      left_join(.,food.ingredient.fit.shelf.variable.food %>% dplyr::select(Food_Category, variablen, Aisle, Shelf,  which.min.out.shelf.food.variable, which.min.value.shelf.food.variable)) %>%
      left_join(.,food.ingredient.fit.shelf.variable %>% dplyr::select(variablen, Aisle, Shelf, which.min.out.shelf.variable, which.min.value.shelf.variable)) %>%
      left_join(.,food.ingredient.fit.aisle.variable.food %>% dplyr::select(Food_Category, variablen, Aisle, which.min.out.aisle.food.variable, which.min.value.aisle.food.variable)) %>%
      left_join(.,food.ingredient.fit.aisle.variable %>% dplyr::select(variablen, Aisle, which.min.out.aisle.variable, which.min.value.aisle.variable)) %>%
      left_join(.,food.ingredient.fit.dep.variable.food %>% dplyr::select(Food_Category, variablen, which.min.out.dep.food.variable = which.min.out.dep.variable.food, which.min.value.dep.food.variable = which.min.value.dep.variable.food)) %>%
      left_join(.,food.ingredient.fit.dep.variable %>% dplyr::select(variablen, which.min.out.dep.variable, which.min.value.dep.variable)) %>%
      left_join(.,food.ingredient.fit.shelf %>% dplyr::select(Aisle, Shelf, which.min.out.shelf, which.min.value.shelf)) %>%
      left_join(.,food.ingredient.fit.aisle %>% dplyr::select(Aisle, which.min.out.aisle, which.min.value.aisle)) %>%
      left_join(.,food.ingredient.fit.dep %>% dplyr::select(Department, which.min.out.dep, which.min.value.dep))
    
    # Looping through columns
    cols.use.estimate = grep('which.min.out',names(dat))
    cols.values.estimate = grep('which.min.value',names(dat))
    
    use.estimates <- rep(NA, nrow(dat))
    update.with <- rep(NA, nrow(dat))
    value.estimate <- rep(NA, nrow(dat))
    threshold.estimate <- rep(NA, nrow(dat))
    # Thresholds
    thresholds <- log2(c(2^.1, 2^.2, 2^.25, 2^.3, 2^.4, 2^.5))
    for(t in thresholds) {
      # looping through columns
      for(ii in cols.values.estimate) {
        update.vals <- which(is.na(use.estimates) & !is.na(dat[,ii]) & dat[,ii] < t)
        use.estimates[update.vals] <- names(dat)[[ii-1]]
        update.with[update.vals] <- dat[update.vals,ii-1]
        value.estimate[update.vals] <- dat[update.vals,ii]
        threshold.estimate[update.vals] <- t
      }
    }
      
    dat <-
      dat %>% 
      mutate(use.estimates = use.estimates,
             value.estimate = value.estimate,
             update.with = update.with,
             threshold.estimate = threshold.estimate) %>% # And updating estimates
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_dif',percent_shelf, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_dif',percent_aisle, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_dif',percent_department, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_dif_all',percent_shelf_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_dif_all',percent_aisle_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_dif_all',percent_department_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_est',percent_estimated_shelf, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_est',percent_estimated_aisle, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_est',percent_estimated_department, percent))
      
    ### Updating salt info
    # Check is:
    #  if food_category recognised as salt
    # and if estimated composition of salt > food packaging, 
    # then update salt content info
    # and set keep to 'YES'
    # else, set keep to 'YES
    # dat <-
    #   dat %>%
    #   left_join(.,salt.dat %>% dplyr::select(id = product_id, salt_amount_g)) %>% # merging in salt dat
    #   mutate(salt_amount_g = ifelse(Food_Category %in% 'Salt', salt_amount_g, NA)) %>% # converting to NA if the ingredient isn't actually salt...
    #   mutate(percent = ifelse(is.na(percent) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%
    #   mutate(percent = ifelse(percent > salt_amount_g & !is.na(salt_amount_g) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%# updating salt percent
    #   mutate(keep_percent = ifelse(Food_Category %in% 'Salt', 'YES',keep_percent)) # and setting keep_percent for salt to 'YES
    
    
    
    # Now updating values based on what is known
    # Getting list of products to interpolate (i.e. any product where total percent != 100) and n ingredients > 1
    product.percent <-
      dat %>%
      group_by(id.new, product_name) %>%
      dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
    
    # merging in
    # And setting order of ingredients in product
    dat <- 
      left_join(dat, product.percent) %>%
      transform(variable = factor(variable, levels = variable.list)) %>%
      mutate(tot_percent = tot_percent_new) %>%
      dplyr::select(-tot_percent_new)
    
    # Ordering
    dat <-
      dat[order(dat$id.new,dat$variable),]
    
    # And list of products to loop through
    product.list <- 
      product.percent$id.new[product.percent$tot_percent_new != 100]
    
    # Looping through these products to update percent composition
    # This is a behemoth of a loop
    for(j in product.list) {
      # Temporary data frame
      tmp.dat <- dat[dat$id.new %in% j,]
      
      # Checking for food additives
      # E.g. vitamins, minerals, presevatives, enzymes, starches, gums, etc
      additive.loc <-
        min(grep('vitamin|mineral|cellulose|\\bstarch|\\bgum|preservative|thiamin|riboflavin|niacin|pyridoxine|cyanocobalamin|pantothenic|biotin|folate|folic acid|choline|carnitine|iron|calcium|zinc|iodine|anti.*stick|enzyme|lactase|stabiliser|stabilizer|emulsifier',
             tmp.dat$value,
             ignore.case = TRUE))
      
      # Automatically setting these to .1% of composition
      if(is.finite(additive.loc) & tmp.dat$keep_percent[additive.loc] %in% 'NO') {
        tmp.dat$percent[additive.loc] <- .1
        tmp.dat$keep_percent[additive.loc] <- 'YES'
        # And exception in case another subsequent ingredient has been identified
        if(additive.loc != nrow(tmp.dat)) {
          if(sum(tmp.dat$keep_percent[(additive.loc+1):nrow(tmp.dat)] %in% 'YES') >= 1 & 
             sum(tmp.dat$percent[(additive.loc+1):nrow(tmp.dat)],na.rm=TRUE) > 0) {
            tmp.dat$percent[additive.loc] <- 
              max(tmp.dat$percent[additive.loc+1:nrow(tmp.dat)], na.rm = TRUE)
          }
        }
      }
      
      # Easy exception
      # If only one "NO" in keep_percent, then that needs to change such that total percent sums to 100
      if(sum(tmp.dat$keep_percent == 'NO') == 1) {
        # If only one ingredient not known, then solve for it using remaining info
        tmp.dat$percent[tmp.dat$keep_percent == 'NO'] <-
          100 - sum(tmp.dat$percent[tmp.dat$keep_percent == 'YES'])
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
      } else if(sum(is.na(tmp.dat$percent)) == nrow(tmp.dat))  {
        # If all no percent is known after above checks, then decreasing power law regression
        tmp.dat$percent = 100/nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (sum(tmp.dat$percent, na.rm = TRUE) == 0) {
        # If estimated percent is 0 after all the checks, then do decreasing power law regression
        tmp.dat$percent = 100 / nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (nrow(tmp.dat) == 1) {
        # If only one row, then percent == 100
        # Unless percent is known from packaging, in which case don't update, and product is later thrown out
        if(tmp.dat$keep_percent %in% 'NO') {
          tmp.dat$percent = 100
          tmp.vector1 <- tmp.dat$percent
          tmp.keep <- tmp.dat$keep_percent
        } else {
          tmp.vector1 <- tmp.dat$percent
          tmp.vector <- tmp.dat$percent
          tmp.keep <- tmp.dat$keep_percent
        }
        # And now interpolating other values
      } else if (nrow(tmp.dat) > 1) {
        # Logic here is to interpolate other values
        # Getting vector of known percents
        tmp.vector <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
        # Get total percent composition of tmp vector
        tot.percent <- sum(tmp.vector, na.rm = TRUE)
        
        ###
        # Interpolate normally
        # If sum tmp vector = 0, then cannot do anything
        # Therefore, if sum does not equal 0
        if(sum(tmp.vector, na.rm = TRUE) != 0) {
          tmp.vector1 = tmp.vector
          
          # Logic checks to fix percent composition of ingredients
          # (1) For example, if N2 == N4, then N3 = N4. Flip tmp.keep to "KEEP"
          # Doing this for every instance this appears
          if(length(tmp.vector1[tmp.keep %in% 'YES']) != length(unique(tmp.vector1[tmp.keep %in% 'YES']))) {
            # Finding location of known composition that are equal
            tmp.table <- 
              table(tmp.vector1[tmp.keep %in% 'YES']) %>%
              .[. != 1] %>% names(.) %>% as.numeric(.)
            
            for(x in 1:length(tmp.table)) {
              tmp.indices <- which(tmp.vector1 == tmp.table[x] & tmp.keep %in% 'YES')
              if(length(tmp.indices) > 0) {
                tmp.vector1[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <-
                  tmp.vector1[tmp.indices[1]]
                tmp.keep[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <- "YES"
              }
            }
            
          }
          
          
          # Getting indices with reported percents
          tmp.check <- which(tmp.vector != 0)
          
          # But can only interpolate if more than 1 ingredient is recognized with a percent
          # if(length(tmp.check) > 1) {
          if(length(tmp.check) > 1) {
            for(k in 1:(length(tmp.check) - 1)) {
              if(!(k %in% c(0))) {
                tmp.vector1[tmp.check[k]:tmp.check[k+1]] <-
                  seq(from = tmp.vector1[tmp.check[k]], to = tmp.vector1[tmp.check[k+1]], length.out = (tmp.check[k+1]-tmp.check[k] + 1)) 
              }
            }
          }
          
          
          # # Updating ingredients for first listed ingredient
          # if(min(tmp.check) != 1) {
          #   tmp.vector1[1:(min(tmp.check)-1)] <-
          #     tmp.vector1[min(tmp.check)]
          # }
        }
        
        ###
        # And exception if total sum of all ingredients > 100 when interpolating normally
        # Only doing this for the last gap in the ingredients list
        # E.g. if ingredients 5-9 are unknown, interpolate these normally
        # But updated interpolation for ingredients 1-3
        # Doing it this way because there is normally a big dropoff in percent composition after the first ingredient or two
        
        
        
        # if(sum(tmp.vector1) > 100) { # If true, then total composition of all ingredients recognized is greater than 100
        #   # Only updating the first gap in the ingredient list
        #   # Assuming that the other graps have been properly managed
        #   
        #   # Removing values that were identified
        #   tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 0
        #   
        #   # Getting sum composition of the other products
        #   # Remaining ingredients need to make this sum to 100
        #   tmp.sum <- sum(tmp.vector1)
        #   
        #   # If only 1 missing ingredient, then all of the missing values are assigned to this
        #   if(length((tmp.check[1] + 1) : (tmp.check[2] -1)) == 1) {
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 100 - tmp.sum
        #   } else { # if more than 1 missing ingredient
        #     # Getting what the interpolation would be
        #     tmp.int <- seq(from = tmp.vector1[tmp.check[1]], to = tmp.vector1[tmp.check[2]], length.out = (tmp.check[2] - tmp.check[1]) + 1)
        #     # Getting only the 2nd to nth values
        #     tmp.int <- tmp.int[2:(length(tmp.int)-1)]
        #     # Now weighting missing ingredients by the values they would be if interpolation was correct
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- # Index of missing values
        #       (100-tmp.sum) * # Total missing percent
        #       tmp.int / # This is what the interpolation would be if the total ingredient list didn't sum to 100
        #       sum(tmp.int) # Then dividing by sum of this interpolation to get percent of missing ingredients
        #   }
        # }
        
        
        ###
        # Updating percentages of last n ingredients after the last ingredient that has a percent recognized
        if(max(tmp.check) < nrow(tmp.dat)) {
          
          tmp.keep[length(tmp.keep)] <- 'YES'
          # Getting how these would interpolate, assuming the 10th ingredient is .1%
          tmp.int <- seq(from = tmp.vector1[tmp.check[length(tmp.check)]], # Value of the last recognized ingredient
                         to = min(c(.1,tmp.vector1[tmp.check[length(tmp.check)]])), # Assuming that the last ingredient is the lesser of 1% or last identified % composition
                         length.out = (nrow(tmp.dat)-tmp.check[length(tmp.check)] + 1)) # Number of ingredients needed
          
          if(length(tmp.int) >= 1) { # Logic check in case there aren't any more ingredients to identify
            # Updating tmp.vector
            tmp.vector1[(tmp.check[length(tmp.check)] + 1) : (tmp.check[length(tmp.check)] + length(tmp.int) - 1)] <-
              tmp.int[2:length(tmp.int)]
          }
        }
        
        
        
        ###
        # Updating percentages or first n ingredients if only one percentage is recognized
        if(min(tmp.check) != 1) { # Logic statement to identify products where the first identified ingredient is not the first
          tmp.sum <- sum(tmp.vector1, na.rm = TRUE) # total composition of ingredients
          tmp.int <- seq(from = (100-tmp.sum), # Percent not yet recognized
                         to = tmp.vector1[min(tmp.check)], # Percent composition of the first product
                         length.out = (min(tmp.check)))
          if(length(tmp.int)>=1) {
            tmp.vector1[1:(min(tmp.check))] <- tmp.int
          }
        }
        
        ###
        # Looping through to correct values where i.e. v1 < v2, etc
        # But only adjusting the non-keep percentages
        for(z in (length(tmp.vector1) - 1):1) {
          if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
            tmp.vector1[z] <- tmp.vector1[z+1]
          }
        }
        
        ###
        # Assuming the last ingredient is correct
        # Need to do this to provide a counterpoint for everything else
        # And also serves as a basis to solve the rest of this puzzle
        
        # First need to check to see if the last ingredient is less than all other ingredients
        if(!is.na(tmp.vector1[length(tmp.vector1)]) & # if the last ingredient isn't the smallest estimated composition
           tmp.vector1[length(tmp.vector1)] != min(tmp.vector1, na.rm = TRUE)) { # then updating this
          tmp.vector1[length(tmp.vector1)] <- min(tmp.vector1, na.rm = TRUE)
        }
        
        # And a few other checks before setting the last ingredient to "YES"
        # This is needed to benchmark the algorithm in case no or limited ingredients are known
        if(sum(tmp.vector1, na.rm = TRUE) > 100) {
          # Dividing unkowns by remaining left needed to sum to 100
          tmp.remaining.check <- 100 - sum(tmp.vector1[tmp.keep %in% 'YES'], na.rm = TRUE)
          # if this results in a negative number, setting the composition of the last ingredient to .1
          if(tmp.remaining.check %in% 0) {
            tmp.vector1[length(tmp.vector1)] <- 0.01
          } else if(tmp.remaining.check > 0) { # This scales the last ingredient proportionately based on what is left for all ingredients and the estimated composition of other ingredients that aren't known
            tmp.vector1[length(tmp.vector1)] <-
              tmp.vector1[length(tmp.vector1)] *
              (tmp.remaining.check / sum(tmp.vector1[tmp.keep %in% 'NO']))
          } else { # Else setting it to .1
            tmp.vector1[length(tmp.vector1)] <- .1
          }
        }
        
        # And now setting it to yes
        tmp.keep[length(tmp.keep)] <- 'YES'
        ###
        # And now correcting percent composition when they do not sum to 100
        # Doing this in a series of while loops
        # Where logic is
        # (a) Adjust the "non-keep" ingredients such that everything sums to 100
        # (b) check that e.g. v1 >= v2, etc
        # (c) check that e.g. v2 >= 100/2
        counter = 1
        while(sum(tmp.vector1) != 100 & counter <= 10) {
          counter = counter + 1
          if(round(sum(tmp.vector1)) != 100) {
            # Getting sum percent of the interpolated ingredients
            tmp.sum.int <- sum(tmp.vector1[tmp.keep == 'NO'])
            tmp.sum <- sum(tmp.vector1[tmp.keep == 'YES'])
            # And scaling so that all ingredients sum to 100
            tmp.vector1[tmp.keep == 'NO'] <-
              tmp.vector1[tmp.keep == 'NO'] *
              ((100 - tmp.sum) / tmp.sum.int)
            
            tmp.vector1[is.nan(tmp.vector1)] <- 0
          } # End if statement
          
          ###
          # Doing two few checks, specifically 
          # (1) %1 > %2 > %3 etc
          # Looping through to correct values where i.e. v1 < v2, etc
          # But only adjusting the non-keep percentages
          for(z in (length(tmp.vector1) - 1):1) {
            if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
              tmp.vector1[z] <- tmp.vector1[z+1]
            }
          }
          
          
          # (2)
          # Updating keep percent when possible
          for(z in (length(tmp.vector1) - 1) : 1) {
            if(tmp.vector1[z] == tmp.vector1[z+1] &
               tmp.keep[z+1] == 'YES') {
              tmp.keep[z] <- 'YES'
            }
          }
          
          # (3) %n msut be less than 100/n
          # But only adjusting the non-identified ingredients
          # And then flagging ones that cannot change again
          for(i in length(tmp.vector1):2) {
            if(tmp.vector1[i] > 100/i & tmp.keep[i] == 'NO') {
              tmp.vector1[i] <- 100/i
            } # End for loop for ingredients
          }
          
        } # End while loop
      } # End of big if else loop
      
      
      
      # updating values in the data frame
      dat$percent[dat$id.new %in% j] <-
        tmp.vector1
      dat$tot_percent[dat$id.new %in% j] <- sum(tmp.vector1, na.rm = TRUE)
    } # End of loop for products
    
    # Dropping id.new
    # dat <- dat %>% dplyr::select(-id.new)
    # Returning data frame
    return(dat)
  } # End function for ingredient interpolating





interpolate.food.ingredients.trial_2 <-
  function(dat) {
    
    # list of variables to loop through
    variable.list = paste0('V',1:length(unique(dat$variable)))
    
    # Some data management
    dat <- 
      dat %>% 
      mutate(percent = percent0) %>%
      mutate(Aisle_Shelf = paste0(Aisle, "_", Shelf)) %>%
      mutate(percent_estimated = NA)
    
    # Saving this to avoid updating averages in the department/aisle/shelf based on already updated averages
    dat.keep <- dat.percent.save %>% mutate(Aisle_Shelf = paste0(Aisle,"_",Shelf)) %>% filter(!is.na(Food_Category))
    
    # Logical sense check for percentages
    for(i in variable.list) { # percent composition cannot be greater than 100/n, where n = order of ingredient in the product
      dat$percent[dat$variable == i & dat$percent > 100/which(variable.list == i) & !is.na(dat$percent)] <- NA
    }
    
    # Creating new column with identified percent composition
    # Identifies rows where percent was listed in back of package information
    dat <-
      dat %>%
      mutate(keep_percent = ifelse(!is.na(percent), 'YES','NO'))
    
    ### Updating salt info
    # Check is:
    #  if food_category recognised as salt
    # and if estimated composition of salt > food packaging, 
    # then update salt content info
    # and set keep to 'YES'
    # else, set keep to 'YES
    dat <-
      dat %>%
      left_join(.,salt.dat %>% dplyr::select(id = product_id, salt_amount_g)) %>% # merging in salt dat
      mutate(salt_check = (100 / variablen)+1) %>% # checking max amount of salt in the product, giving a bit of breathing space for rounding errors
      mutate(salt_amount_g = ifelse(salt_amount_g > salt_check, salt_amount_g / 1000, salt_amount_g)) %>% # updating max amount of salt if it is incorrect - this will be a units issue
      mutate(salt_amount_g = ifelse(Food_Category %in% 'Salt', salt_amount_g, NA)) %>% # converting to NA if the ingredient isn't actually salt...
      mutate(percent = ifelse(is.na(percent) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%
      mutate(percent = ifelse(percent > salt_amount_g & !is.na(salt_amount_g) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>% # updating salt percent
      mutate(keep_percent = ifelse(Food_Category %in% 'Salt' & !is.na(percent), 'YES',keep_percent)) %>% # and setting keep_percent for salt to 'YES
      dplyr::select(-salt_check) # and dropping column
    
    # updating max composition for additives, vitamins, and minerals, to 1% of product
    dat <- 
      dat %>%
      mutate(percent_check = 100 / variablen + .5) %>% # max percent, with some leeway for rounding errors
      mutate(percent = ifelse(Food_Category %in% c('Other Food Additive','Vitamin','Mineral') & is.na(percent),1,percent)) %>%
      mutate(percent = ifelse(percent > percent_check & Food_Category %in% c('Other Food Additive','Vitamin','Mineral'), percent_check, percent)) %>%
      mutate(keep_percent = ifelse(!is.na(percent) & Food_Category %in% c('Other Food Additive','Vitamin','Mineral'), 'YES', keep_percent))
    
    # View(dat %>% dplyr::select(product_name, value, variablen, Food_Category,percent,salt_amount_g,keep_percent))
    
    # And doing checks on number of ingredients in each food category, sub category, and sub sub category
    # The general rule is that if there are >= 10 observations, then use info from the sub sub category, if not, use info from the sub cateogry, and if not, use info from the original category
    # Getting this aggregation by shelf, aisle, and department
    
    # By department
    count.dep <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Food_Category) %>%
                  dplyr::summarise(count_dep_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_dep_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_dep_sub_sub = n())) %>%
      filter(!is.na(Food_Category))  %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_dep_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_dep_sub)) %>%
      mutate(count_dep_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_dep_sub_sub))
    # By aisle
    count.ais <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Aisle, Food_Category) %>%
                  dplyr::summarise(count_ais_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Aisle, Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_ais_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Aisle, Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_ais_sub_sub = n())) %>%
      filter(!is.na(Food_Category)) %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_ais_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_ais_sub)) %>%
      mutate(count_ais_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_ais_sub_sub))
    
    # By shelf
    count.shelf <-
      left_join(dat %>% 
                  filter(!is.na(Food_Category)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category) %>%
                  dplyr::summarise(count_shelf_main = n()),
                dat %>% 
                  filter(!is.na(Food_Category_sub)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category, Food_Category_sub) %>%
                  dplyr::summarise(count_shelf_sub = n())) %>%
      left_join(.,
                dat %>% 
                  filter(!is.na(Food_Category_sub_sub)) %>%
                  dplyr::group_by(Aisle, Shelf, Food_Category, Food_Category_sub, Food_Category_sub_sub) %>%
                  dplyr::summarise(count_shelf_sub_sub = n())) %>%
      filter(!is.na(Food_Category)) %>% # Not updating count for pork, beef, milk, and eggs - the sub categoreis on these are e.g. free range, etc
      mutate(count_shelf_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_shelf_sub)) %>%
      mutate(count_shelf_sub_sub = ifelse(grepl('pig|bovine|milk|egg',Food_Category,ignore.case=TRUE),0,count_shelf_sub_sub))
    
    
    # Estimating percent composition using info from similar products and the same food category ----
    # Updating values based on shelf first, if more than 10 products in that shelf
    # But only if NAs
    shelf.sum <-
      dat.keep %>%
      left_join(.,count.shelf) %>%
      mutate(Food_Category_sub = 
               ifelse(count_shelf_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_shelf_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department, Aisle, Shelf,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Shelf, Food_Category, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shelf.sum %>% filter(!is.na(Food_Category)))
    
    # Updating values based on aisle, if more than 10 products in that aisle
    aisle.sum <-
      dat.keep %>%
      left_join(.,count.ais) %>%
      mutate(Food_Category_sub = 
               ifelse(count_ais_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_ais_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department, Aisle,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Aisle, Food_Category, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable)), NA, percent_aisle))
    
    dat <-
      left_join(dat, aisle.sum %>% filter(!is.na(Food_Category)))
    
    # Updating values based on department
    department.sum <-
      dat.keep %>%
      left_join(.,count.dep) %>%
      mutate(Food_Category_sub = 
               ifelse(count_dep_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_dep_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department,
                    variable, variablen, value, Food_Category, percent) %>%
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, Food_Category, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department))
    
    dat <-
      left_join(dat, department.sum %>% filter(!is.na(Food_Category)))
    
    # Repeating the above across all food categories ----
    shel.sum <-
      dat.keep %>%
      group_by(Aisle, Shelf, variable) %>%
      dplyr::summarise(percent_shelf = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_shelf = ifelse(is.nan(percent_shelf),NA,percent_shelf)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_shelf = ifelse(percent_shelf > 100/as.numeric(gsub('V','',variable)), NA, percent_shelf))
    
    dat <-
      left_join(dat, shel.sum %>% dplyr::rename(percent_shelf_all = percent_shelf))
    
    
    ais.sum <-
      dat.keep %>%
      group_by(Aisle, variable) %>%
      dplyr::summarise(percent_aisle = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_aisle = ifelse(is.nan(percent_aisle),NA,percent_aisle)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_aisle = ifelse(percent_aisle > 100/as.numeric(gsub('V','',variable)), NA, percent_aisle))
    
    dat <-
      left_join(dat, ais.sum %>% dplyr::rename(percent_aisle_all = percent_aisle))
    
    dep.sum <- 
      dat.keep %>%
      group_by(Department, variable) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = dplyr::n()) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      filter(count >= 10) %>% dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department))
    
    dat <-
      left_join(dat, dep.sum %>% dplyr::rename(percent_department_all = percent_department))
    
    # Now interpolating based on power law regression equations ----
    # Doing this by building regression models for each Aisle and Shelf
    # Structure of model is percent = 1/variable^2, w/ no intercept to avoid negative values
    # Where variable is the location of variable in the product's ingredient list
    # Starting with each shelf
    # List to loop
    loop.list <- unique(dat.keep$Aisle_Shelf)
    dat$percent_estimated_shelf = NA
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        left_join(.,count.shelf) %>%
        mutate(Food_Category_sub = 
                 ifelse(count_shelf_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
        mutate(Food_Category = 
                 ifelse(count_shelf_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
        dplyr::select(id, product_name, Retailer, Department, Aisle, Shelf,
                      variable, variablen, value, Food_Category, percent, Aisle_Shelf) %>%
        filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
        filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
        unique(.) %>%
        filter(Aisle_Shelf %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z & dat.keep$variablen <= 15]) 
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle_Shelf %in% z]) >= 5) {
        # tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        
        lm.list <-
          list(lm(tmp.dat$percent ~ I(1/tmp.dat$variablen^2) - 1),
               lm(tmp.dat$percent ~ I(1/tmp.dat$variablen) - 1),
               lm(tmp.dat$percent ~ tmp.dat$variablen))# No intercept
        r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
        # p.list <- c(summary(lm.list[[1]])$coefficients[4],summary(lm.list[[2]])$coefficients[4])
        
        tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
        # Only updating if adj R^2 > .75 and P < .05
        # Second check to get rid of nan statements
        if(!is.nan(summary(tmp.lm)$adj.r.squared)) {
          if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
            if(which(r2.list %in% max(r2.list)) %in% 1) {
              dat <-
                dat %>%
                mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                        tmp.lm$coefficients[[1]] / variablen^2,
                                                        percent_estimated))
            } else if (which(r2.list %in% max(r2.list)) %in% 2) {
              dat <-
                dat %>%
                mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                        tmp.lm$coefficients[[1]] / variablen,
                                                        percent_estimated))
            } else {
              dat <-
                dat %>%
                mutate(percent_estimated_shelf = ifelse(Aisle_Shelf %in% z,
                                                        tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen,
                                                        percent_estimated))
            }
            
          }
        }
      }
    }
    
    # Repeating for Aisles
    # Setting percent_estimated back to 0
    dat <-
      dat %>%
      mutate(percent_estimated_aisle = NA)
    # List to loop
    loop.list <- unique(dat.keep$Aisle)
    
    # Looping
    for(z in loop.list) {
      # Tmp data frame
      tmp.dat <- 
        dat.keep %>%
        left_join(.,count.ais) %>%
        mutate(Food_Category_sub = 
                 ifelse(count_ais_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
        mutate(Food_Category = 
                 ifelse(count_ais_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
        dplyr::select(id, product_name, Retailer, Department, Aisle,
                      variable, variablen, value, Food_Category, percent) %>%
        filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
        filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
        unique(.) %>%
        filter(Aisle %in% z) %>%
        filter(!is.na(percent))
      # Proportion of variables represented
      prop_variables = length(unique(tmp.dat$variablen)) / max(dat.keep$variablen[dat.keep$Aisle %in% z & dat.keep$variablen <= 15])
      
      # Only building if decent data is there
      # E.g. don't want to interpolate with bad data
      if(prop_variables >= 0.5 & max(dat.keep$variablen[dat.keep$Aisle %in% z]) >= 5) {
        # tmp.lm = lm(tmp.dat$percent ~I(1/tmp.dat$variablen^2) - 1)
        
        lm.list <-
          list(lm(tmp.dat$percent ~ I(1/tmp.dat$variablen^2) - 1),
               lm(tmp.dat$percent ~ I(1/tmp.dat$variablen) - 1),
               lm(tmp.dat$percent ~ tmp.dat$variablen))# No intercept
        r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
        tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
        # Only updating if adj R^2 > .5 and P < .05
        if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
          if(which(r2.list %in% max(r2.list)) %in% 1) {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen^2,
                                                      percent_estimated))
          } else if (which(r2.list %in% max(r2.list)) %in% 2) {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] / variablen,
                                                      percent_estimated))
          } else {
            dat <-
              dat %>%
              mutate(percent_estimated_aisle = ifelse(Aisle %in% z,
                                                      tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen,
                                                      percent_estimated))
          }
          
          
        }
      }
    }
    
    # Repeating for Departments
    # Don't need a loop, because this already loops across departments
    
    
    # Repeating for departments
    dep.sum <- 
      dat.keep %>%
      left_join(.,count.dep) %>%
      mutate(Food_Category_sub = 
               ifelse(count_dep_sub_sub >= 10 & !is.na(Food_Category_sub_sub), Food_Category_sub_sub, Food_Category_sub)) %>%
      mutate(Food_Category = 
               ifelse(count_dep_sub >= 10 & !is.na(Food_Category_sub), Food_Category_sub, Food_Category)) %>%
      dplyr::select(id, product_name, Retailer, Department,
                    variable, variablen, value, Food_Category, percent) %>%
      filter(!is.na(Food_Category)) %>% # Dropping products not recognised as food
      filter(variablen <= 15) %>% # only keeping first 15 ingredients, after this composition becomes nominal
      unique(.) %>%
      mutate(count = 1) %>%
      mutate(count = ifelse(percent %in% 0 | is.na(percent), 0 , 1)) %>%
      group_by(Department, variable, variablen) %>%
      dplyr::summarise(percent_department = mean(percent, na.rm = TRUE),
                       count = sum(count)) %>%
      mutate(percent_department = ifelse(is.nan(percent_department),NA,percent_department)) %>%
      mutate(percent_department = ifelse(count < 10, NA, percent_department)) %>% 
      dplyr::select(-count) %>%
      mutate(percent_department = ifelse(percent_department > 100/as.numeric(gsub('V','',variable)), NA, percent_department)) %>%
      mutate(variablen = as.numeric(gsub("V","",variable)))
    
    prop_variables = length(unique(dep.sum$variablen[!is.na(dep.sum$percent_department)])) / max(dat.keep$variablen[dat.keep$variablen <= 15])
    
    dat$percent_estimated_department = NA
    
    # Building regression to estimate composition of other ingredients
    if(prop_variables >= 0.5 & max(dep.sum$variablen) >= 5) {
      lm.list <-
        list(lm(dat.keep$percent ~ I(1/dat.keep$variablen^2) - 1),
             lm(dat.keep$percent ~ I(1/dat.keep$variablen) - 1),
             lm(dat.keep$percent ~ dat.keep$variablen))# No intercept
      r2.list <- c(summary(lm.list[[1]])$adj.r.squared,summary(lm.list[[2]])$adj.r.squared,summary(lm.list[[3]])$adj.r.squared)
      # p.list <- c(summary(lm.list[[1]])$coefficients[4],summary(lm.list[[2]])$coefficients[4])
      
      tmp.lm <- lm.list[[which(r2.list %in% max(r2.list))]]
      
      # tmp.lm = lm(dat.keep$percent ~ I(1/dat.keep$variablen) - 1) # No intercept
      # Predicting
      if(summary(tmp.lm)$adj.r.squared > 0.75 & summary(tmp.lm)$coefficients[nrow(summary(tmp.lm)$coefficients),4] < 0.05) {
        # dep.sum <-
        #   dep.sum %>%
        #   mutate(percent_estimated = tmp.lm$coefficients[[1]] / variablen^2) %>% # Predicting 
        #   mutate(percent_department = ifelse(is.na(percent_department), percent_estimated, percent_department)) %>%
        #   dplyr::select(variable, percent_department)
        # Merging back in
        if(which(r2.list %in% max(r2.list)) %in% 1) {
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] / variablen^2) 
        } else if (which(r2.list %in% max(r2.list)) %in% 2 ){
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] / variablen)
        } else {
          dat <-
            dat %>%
            mutate(percent_estimated_department = tmp.lm$coefficients[[1]] + tmp.lm$coefficients[[2]] * variablen) %>%
            mutate(check = predict(tmp.lm,dat))
        }
        
      }
    }
    
    # Calculating absolute difference to use for a given food category and ingredient number
    dat <-
      dat %>%
      mutate(abs_difference = NA) %>%
      mutate(shelf_dif = abs(log2(percent / percent_shelf)),
             aisle_dif = abs(log2(percent / percent_aisle)),
             dep_dif = abs(log2(percent / percent_department)),
             shelf_dif_all = abs(log2(percent / percent_shelf_all)),
             aisle_dif_all = abs(log2(percent / percent_aisle_all)),
             dep_dif_all = abs(log2(percent / percent_aisle_all)),
             shelf_dif_estimated = abs(log2(percent / percent_estimated_shelf)),
             aisle_dif_estimated = abs(log2(percent / percent_estimated_aisle)),
             dep_dif_estimated = abs(log2(percent / percent_estimated_department)))
    
    ### Now getting best average fit for:
    # And assigning which approach to use based on the below ordering
    # E.g. if data on (1) is available, use that, if not, use data from (2), if not, then data from (3), etc
    # With additional quality checks - e.g. log2 ratio < .2
    # (1) food category by ingredient number by aisle and shelf
    # (2) ingredient number by aisle and shelf
    # (3) food category by ingredient number by aisle
    # (4) ingredient number by aisle
    # (5) food category by ingredient number by department
    # (6) ingredient number by department
    # (7) shelf
    # (8) aisle
    # (9) department
    
    ### Shelf by ingredient
    food.ingredient.fit.shelf.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen, Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_dif = mean(shelf_dif,na.rm=TRUE),
                       aisle_dif = mean(aisle_dif,na.rm=TRUE),
                       dep_dif = mean(dep_dif,na.rm=TRUE),
                       shelf_dif_all = mean(shelf_dif_all,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf.variable.food[ii,c(which(names(food.ingredient.fit.shelf.variable.food) %in% 'shelf_dif'):which(names(food.ingredient.fit.shelf.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf.variable.food[ii,c(which(names(food.ingredient.fit.shelf.variable.food) %in% 'shelf_dif'):which(names(food.ingredient.fit.shelf.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf.variable.food$which.min.value.shelf.food.variable = which.min.value
    food.ingredient.fit.shelf.variable.food$which.min.out.shelf.food.variable = names(which.min.out)
    
    # And getting rid of NAs
    food.ingredient.fit.shelf.variable.food <-
      food.ingredient.fit.shelf.variable.food %>%
      mutate(which.min.out.shelf.food.variable = ifelse(which.min.value.shelf.food.variable %in% 99999, NA, which.min.out.shelf.food.variable),
             which.min.value.shelf.food.variable = ifelse(which.min.value.shelf.food.variable %in% 99999, NA, which.min.value.shelf.food.variable),)
    
    ### Shelf by ingredient number
    food.ingredient.fit.shelf.variable <- 
      dat %>% 
      group_by(variablen, Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_dif_all = mean(shelf_dif_all,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf.variable[ii,c(which(names(food.ingredient.fit.shelf.variable) %in% 'shelf_dif_all'):which(names(food.ingredient.fit.shelf.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf.variable[ii,c(which(names(food.ingredient.fit.shelf.variable) %in% 'shelf_dif_all'):which(names(food.ingredient.fit.shelf.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf.variable$which.min.value.shelf.variable = which.min.value
    food.ingredient.fit.shelf.variable$which.min.out.shelf.variable = names(which.min.out)
    
    food.ingredient.fit.shelf.variable <-
      food.ingredient.fit.shelf.variable %>%
      mutate(which.min.out.shelf.variable = ifelse(which.min.value.shelf.variable %in% 99999, NA, which.min.out.shelf.variable),
             which.min.value.shelf.variable = ifelse(which.min.value.shelf.variable %in% 99999, NA, which.min.value.shelf.variable),)
    
    ### Aisle by ingredient and food
    food.ingredient.fit.aisle.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen, Aisle) %>% # Getting averages
      dplyr::summarise(aisle_dif = mean(aisle_dif,na.rm=TRUE),
                       dep_dif = mean(dep_dif,na.rm=TRUE),
                       aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle.variable.food[ii,c(which(names(food.ingredient.fit.aisle.variable.food) %in% 'aisle_dif'):which(names(food.ingredient.fit.aisle.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle.variable.food[ii,c(which(names(food.ingredient.fit.aisle.variable.food) %in% 'aisle_dif'):which(names(food.ingredient.fit.aisle.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle.variable.food$which.min.value.aisle.food.variable = which.min.value
    food.ingredient.fit.aisle.variable.food$which.min.out.aisle.food.variable = names(which.min.out)
    # And updating NAs
    food.ingredient.fit.aisle.variable.food <-
      food.ingredient.fit.aisle.variable.food %>%
      mutate(which.min.out.aisle.food.variable = ifelse(which.min.value.aisle.food.variable %in% 99999, NA, which.min.out.aisle.food.variable),
             which.min.value.aisle.food.variable = ifelse(which.min.value.aisle.food.variable %in% 99999, NA, which.min.value.aisle.food.variable),)
    
    ### Aisle by ingredient number
    food.ingredient.fit.aisle.variable <- 
      dat %>% 
      group_by(variablen, Aisle) %>% # Getting averages
      dplyr::summarise(aisle_dif_all = mean(aisle_dif_all,na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle.variable[ii,c(which(names(food.ingredient.fit.aisle.variable) %in% 'aisle_dif_all'):which(names(food.ingredient.fit.aisle.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle.variable[ii,c(which(names(food.ingredient.fit.aisle.variable) %in% 'aisle_dif_all'):which(names(food.ingredient.fit.aisle.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle.variable$which.min.value.aisle.variable = which.min.value
    food.ingredient.fit.aisle.variable$which.min.out.aisle.variable = names(which.min.out)
    # And updating NAs
    food.ingredient.fit.aisle.variable <-
      food.ingredient.fit.aisle.variable %>%
      mutate(which.min.out.aisle.variable = ifelse(which.min.value.aisle.variable %in% 99999, NA, which.min.out.aisle.variable),
             which.min.value.aisle.variable = ifelse(which.min.value.aisle.variable %in% 99999, NA, which.min.value.aisle.variable),)
    
    ### Department by ingredient and food
    food.ingredient.fit.dep.variable.food <- 
      dat %>% 
      group_by(Food_Category, variablen) %>% # Getting averages
      dplyr::summarise(dep_dif = mean(dep_dif, na.rm=TRUE),
                       dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.dep.variable.food)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.dep.variable.food[ii,c(which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_dif'):which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.dep.variable.food[ii,c(which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_dif'):which(names(food.ingredient.fit.dep.variable.food) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.dep.variable.food$which.min.value.dep.variable.food = which.min.value
    food.ingredient.fit.dep.variable.food$which.min.out.dep.variable.food = names(which.min.out)
    
    # And updating NAs
    food.ingredient.fit.dep.variable.food <-
      food.ingredient.fit.dep.variable.food %>%
      mutate(which.min.out.dep.variable.food = ifelse(which.min.value.dep.variable.food %in% 99999, NA, which.min.out.dep.variable.food),
             which.min.value.dep.variable.food = ifelse(which.min.value.dep.variable.food %in% 99999, NA, which.min.value.dep.variable.food),)
    
    ### Department by ingredient number
    food.ingredient.fit.dep.variable <- 
      dat %>% 
      group_by(variablen) %>% # Getting averages
      dplyr::summarise(dep_dif_all = mean(dep_dif_all,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.dep.variable)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.dep.variable[ii,c(which(names(food.ingredient.fit.dep.variable) %in% 'dep_dif_all'):which(names(food.ingredient.fit.dep.variable) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.dep.variable[ii,c(which(names(food.ingredient.fit.dep.variable) %in% 'dep_dif_all'):which(names(food.ingredient.fit.dep.variable) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.dep.variable$which.min.value.dep.variable = which.min.value
    food.ingredient.fit.dep.variable$which.min.out.dep.variable = names(which.min.out)
    # Updating NAs
    food.ingredient.fit.dep.variable <-
      food.ingredient.fit.dep.variable %>%
      mutate(which.min.out.dep.variable = ifelse(which.min.value.dep.variable %in% 99999, NA, which.min.out.dep.variable),
             which.min.value.dep.variable = ifelse(which.min.value.dep.variable %in% 99999, NA, which.min.value.dep.variable),)
    
    ### By Shelf
    food.ingredient.fit.shelf <- 
      dat %>% 
      group_by(Aisle, Shelf) %>% # Getting averages
      dplyr::summarise(shelf_est = mean(shelf_dif_estimated,na.rm=TRUE),
                       aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.shelf)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.shelf[ii,c(which(names(food.ingredient.fit.shelf) %in% 'shelf_est'):which(names(food.ingredient.fit.shelf) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.shelf[ii,c(which(names(food.ingredient.fit.shelf) %in% 'shelf_est'):which(names(food.ingredient.fit.shelf) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.shelf$which.min.value.shelf = which.min.value
    food.ingredient.fit.shelf$which.min.out.shelf = names(which.min.out)
    # Updating NAs
    food.ingredient.fit.shelf <-
      food.ingredient.fit.shelf %>%
      mutate(which.min.out.shelf = ifelse(which.min.value.shelf %in% 99999, NA, which.min.out.shelf),
             which.min.value.shelf = ifelse(which.min.value.shelf %in% 99999, NA, which.min.value.shelf),)
    ### By Aisle
    food.ingredient.fit.aisle <- 
      dat %>% 
      group_by(Aisle) %>% # Getting averages
      dplyr::summarise(aisle_est = mean(aisle_dif_estimated,na.rm=TRUE),
                       dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(dep_est = ifelse(is.na(dep_est),99999,dep_est))
    # Getting most accurate estimate
    which.min.out <- c()
    which.min.value <- c()
    for(ii in 1:nrow(food.ingredient.fit.aisle)) {
      which.min.out <- c(which.min.out, which.min(food.ingredient.fit.aisle[ii,c(which(names(food.ingredient.fit.aisle) %in% 'aisle_est'):which(names(food.ingredient.fit.aisle) %in% 'dep_est'))]))
      which.min.value <- c(which.min.value, min(food.ingredient.fit.aisle[ii,c(which(names(food.ingredient.fit.aisle) %in% 'aisle_est'):which(names(food.ingredient.fit.aisle) %in% 'dep_est'))], na.rm = TRUE))
    }
    # And updating
    food.ingredient.fit.aisle$which.min.value.aisle = which.min.value
    food.ingredient.fit.aisle$which.min.out.aisle = names(which.min.out)
    # Removing NAs
    food.ingredient.fit.aisle <-
      food.ingredient.fit.aisle %>%
      mutate(which.min.out.aisle = ifelse(which.min.value.aisle %in% 99999, NA, which.min.out.aisle),
             which.min.value.aisle = ifelse(which.min.value.aisle %in% 99999, NA, which.min.value.aisle),)
    
    ### By Department
    food.ingredient.fit.dep <- 
      dat %>% 
      group_by(Department) %>% # Getting averages
      dplyr::summarise(dep_est = mean(dep_dif_estimated,na.rm=TRUE)) %>%
      mutate(which.min.value.dep = 'dep_est',
             which.min.out.dep = dep_est)
    
    ### Merging
    # To decide which approach to use
    # Go down order as described above
    # Whilst throwing in some quality checks (first checking if log ratio 2^.1, then 2^.25, and then 2^.5)
    # If none of the above, then estimating composition of that ingredient by interpolating
    dat <-
      dat %>% # Merging
      left_join(.,food.ingredient.fit.shelf.variable.food %>% dplyr::select(Food_Category, variablen, Aisle, Shelf,  which.min.out.shelf.food.variable, which.min.value.shelf.food.variable)) %>%
      left_join(.,food.ingredient.fit.shelf.variable %>% dplyr::select(variablen, Aisle, Shelf, which.min.out.shelf.variable, which.min.value.shelf.variable)) %>%
      left_join(.,food.ingredient.fit.aisle.variable.food %>% dplyr::select(Food_Category, variablen, Aisle, which.min.out.aisle.food.variable, which.min.value.aisle.food.variable)) %>%
      left_join(.,food.ingredient.fit.aisle.variable %>% dplyr::select(variablen, Aisle, which.min.out.aisle.variable, which.min.value.aisle.variable)) %>%
      left_join(.,food.ingredient.fit.dep.variable.food %>% dplyr::select(Food_Category, variablen, which.min.out.dep.food.variable = which.min.out.dep.variable.food, which.min.value.dep.food.variable = which.min.value.dep.variable.food)) %>%
      left_join(.,food.ingredient.fit.dep.variable %>% dplyr::select(variablen, which.min.out.dep.variable, which.min.value.dep.variable)) %>%
      left_join(.,food.ingredient.fit.shelf %>% dplyr::select(Aisle, Shelf, which.min.out.shelf, which.min.value.shelf)) %>%
      left_join(.,food.ingredient.fit.aisle %>% dplyr::select(Aisle, which.min.out.aisle, which.min.value.aisle)) %>%
      left_join(.,food.ingredient.fit.dep %>% dplyr::select(Department, which.min.out.dep, which.min.value.dep))
    
    # Looping through columns
    cols.use.estimate = grep('which.min.out',names(dat))
    cols.values.estimate = grep('which.min.value',names(dat))
    
    use.estimates <- rep(NA, nrow(dat))
    update.with <- rep(NA, nrow(dat))
    value.estimate <- rep(NA, nrow(dat))
    threshold.estimate <- rep(NA, nrow(dat))
    # Thresholds
    thresholds <- log2(c(2^.1, 2^.2, 2^.25, 2^.3, 2^.4, 2^.5))
    for(t in thresholds) {
      # looping through columns
      for(ii in cols.values.estimate) {
        update.vals <- which(is.na(use.estimates) & !is.na(dat[,ii]) & dat[,ii] < t)
        use.estimates[update.vals] <- names(dat)[[ii-1]]
        update.with[update.vals] <- dat[update.vals,ii-1]
        value.estimate[update.vals] <- dat[update.vals,ii]
        threshold.estimate[update.vals] <- t
      }
    }
    
    dat <-
      dat %>% 
      mutate(use.estimates = use.estimates,
             value.estimate = value.estimate,
             update.with = update.with,
             threshold.estimate = threshold.estimate) %>% # And updating estimates
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_dif',percent_shelf, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_dif',percent_aisle, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_dif',percent_department, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_dif_all',percent_shelf_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_dif_all',percent_aisle_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_dif_all',percent_department_all, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'shelf_est',percent_estimated_shelf, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'aisle_est',percent_estimated_aisle, percent)) %>%
      mutate(percent = ifelse(is.na(percent) & update.with %in% 'dep_est',percent_estimated_department, percent))
    
    ### Updating salt info
    # Check is:
    #  if food_category recognised as salt
    # and if estimated composition of salt > food packaging, 
    # then update salt content info
    # and set keep to 'YES'
    # else, set keep to 'YES
    # dat <-
    #   dat %>%
    #   left_join(.,salt.dat %>% dplyr::select(id = product_id, salt_amount_g)) %>% # merging in salt dat
    #   mutate(salt_amount_g = ifelse(Food_Category %in% 'Salt', salt_amount_g, NA)) %>% # converting to NA if the ingredient isn't actually salt...
    #   mutate(percent = ifelse(is.na(percent) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%
    #   mutate(percent = ifelse(percent > salt_amount_g & !is.na(salt_amount_g) & Food_Category %in% 'Salt',salt_amount_g, percent)) %>%# updating salt percent
    #   mutate(keep_percent = ifelse(Food_Category %in% 'Salt', 'YES',keep_percent)) # and setting keep_percent for salt to 'YES
    
    
    
    # Now updating values based on what is known
    # Getting list of products to interpolate (i.e. any product where total percent != 100) and n ingredients > 1
    product.percent <-
      dat %>%
      group_by(id.new, product_name) %>%
      dplyr::summarise(tot_percent_new = sum(percent, na.rm = TRUE)) 
    
    # merging in
    # And setting order of ingredients in product
    dat <- 
      left_join(dat, product.percent) %>%
      transform(variable = factor(variable, levels = variable.list)) %>%
      mutate(tot_percent = tot_percent_new) %>%
      dplyr::select(-tot_percent_new)
    
    # Ordering
    dat <-
      dat[order(dat$id.new,dat$variable),]
    
    # And list of products to loop through
    product.list <- 
      product.percent$id.new[product.percent$tot_percent_new != 100]
    
    # Looping through these products to update percent composition
    # This is a behemoth of a loop
    for(j in product.list) {
      # Temporary data frame
      tmp.dat <- dat[dat$id.new %in% j,]
      
      # Checking for food additives
      # E.g. vitamins, minerals, presevatives, enzymes, starches, gums, etc
      additive.loc <-
        min(grep('vitamin|mineral|cellulose|\\bstarch|\\bgum|preservative|thiamin|riboflavin|niacin|pyridoxine|cyanocobalamin|pantothenic|biotin|folate|folic acid|choline|carnitine|iron|calcium|zinc|iodine|anti.*stick|enzyme|lactase|stabiliser|stabilizer|emulsifier',
                 tmp.dat$value,
                 ignore.case = TRUE))
      
      # Automatically setting these to .1% of composition
      if(is.finite(additive.loc) & tmp.dat$keep_percent[additive.loc] %in% 'NO') {
        tmp.dat$percent[additive.loc] <- .1
        tmp.dat$keep_percent[additive.loc] <- 'YES'
        # And exception in case another subsequent ingredient has been identified
        if(additive.loc != nrow(tmp.dat)) {
          if(sum(tmp.dat$keep_percent[(additive.loc+1):nrow(tmp.dat)] %in% 'YES') >= 1 & 
             sum(tmp.dat$percent[(additive.loc+1):nrow(tmp.dat)],na.rm=TRUE) > 0) {
            tmp.dat$percent[additive.loc] <- 
              max(tmp.dat$percent[additive.loc+1:nrow(tmp.dat)], na.rm = TRUE)
          }
        }
      }
      
      # Easy exception
      # If only one "NO" in keep_percent, then that needs to change such that total percent sums to 100
      if(sum(tmp.dat$keep_percent == 'NO') == 1) {
        # If only one ingredient not known, then solve for it using remaining info
        tmp.dat$percent[tmp.dat$keep_percent == 'NO'] <-
          100 - sum(tmp.dat$percent[tmp.dat$keep_percent == 'YES'])
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
      } else if(sum(is.na(tmp.dat$percent)) == nrow(tmp.dat))  {
        # If all no percent is known after above checks, then decreasing power law regression
        tmp.dat$percent = 100/nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (sum(tmp.dat$percent, na.rm = TRUE) == 0) {
        # If estimated percent is 0 after all the checks, then do decreasing power law regression
        tmp.dat$percent = 100 / nrow(tmp.dat)
        tmp.vector1 <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
      } else if (nrow(tmp.dat) == 1) {
        # If only one row, then percent == 100
        # Unless percent is known from packaging, in which case don't update, and product is later thrown out
        if(tmp.dat$keep_percent %in% 'NO') {
          tmp.dat$percent = 100
          tmp.vector1 <- tmp.dat$percent
          tmp.keep <- tmp.dat$keep_percent
        } else {
          tmp.vector1 <- tmp.dat$percent
          tmp.vector <- tmp.dat$percent
          tmp.keep <- tmp.dat$keep_percent
        }
        # And now interpolating other values
      } else if (nrow(tmp.dat) > 1) {
        # Logic here is to interpolate other values
        # Getting vector of known percents
        tmp.vector <- tmp.dat$percent
        tmp.keep <- tmp.dat$keep_percent
        
        # Get total percent composition of tmp vector
        tot.percent <- sum(tmp.vector, na.rm = TRUE)
        
        ###
        # Interpolate normally
        # If sum tmp vector = 0, then cannot do anything
        # Therefore, if sum does not equal 0
        if(sum(tmp.vector, na.rm = TRUE) != 0) {
          tmp.vector1 = tmp.vector
          
          # Logic checks to fix percent composition of ingredients
          # (1) For example, if N2 == N4, then N3 = N4. Flip tmp.keep to "KEEP"
          # Doing this for every instance this appears
          if(length(tmp.vector1[tmp.keep %in% 'YES']) != length(unique(tmp.vector1[tmp.keep %in% 'YES']))) {
            # Finding location of known composition that are equal
            tmp.table <- 
              table(tmp.vector1[tmp.keep %in% 'YES']) %>%
              .[. != 1] %>% names(.) %>% as.numeric(.)
            
            for(x in 1:length(tmp.table)) {
              tmp.indices <- which(tmp.vector1 == tmp.table[x] & tmp.keep %in% 'YES')
              if(length(tmp.indices) > 0) {
                tmp.vector1[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <-
                  tmp.vector1[tmp.indices[1]]
                tmp.keep[tmp.indices[1] : tmp.indices[length(tmp.indices)]] <- "YES"
              }
            }
            
          }
          
          
          # Getting indices with reported percents
          tmp.check <- which(tmp.vector != 0)
          
          # But can only interpolate if more than 1 ingredient is recognized with a percent
          # if(length(tmp.check) > 1) {
          if(length(tmp.check) > 1) {
            for(k in 1:(length(tmp.check) - 1)) {
              if(!(k %in% c(0))) {
                tmp.vector1[tmp.check[k]:tmp.check[k+1]] <-
                  seq(from = tmp.vector1[tmp.check[k]], to = tmp.vector1[tmp.check[k+1]], length.out = (tmp.check[k+1]-tmp.check[k] + 1)) 
              }
            }
          }
          
          
          # # Updating ingredients for first listed ingredient
          # if(min(tmp.check) != 1) {
          #   tmp.vector1[1:(min(tmp.check)-1)] <-
          #     tmp.vector1[min(tmp.check)]
          # }
        }
        
        ###
        # And exception if total sum of all ingredients > 100 when interpolating normally
        # Only doing this for the last gap in the ingredients list
        # E.g. if ingredients 5-9 are unknown, interpolate these normally
        # But updated interpolation for ingredients 1-3
        # Doing it this way because there is normally a big dropoff in percent composition after the first ingredient or two
        
        
        
        # if(sum(tmp.vector1) > 100) { # If true, then total composition of all ingredients recognized is greater than 100
        #   # Only updating the first gap in the ingredient list
        #   # Assuming that the other graps have been properly managed
        #   
        #   # Removing values that were identified
        #   tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 0
        #   
        #   # Getting sum composition of the other products
        #   # Remaining ingredients need to make this sum to 100
        #   tmp.sum <- sum(tmp.vector1)
        #   
        #   # If only 1 missing ingredient, then all of the missing values are assigned to this
        #   if(length((tmp.check[1] + 1) : (tmp.check[2] -1)) == 1) {
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- 100 - tmp.sum
        #   } else { # if more than 1 missing ingredient
        #     # Getting what the interpolation would be
        #     tmp.int <- seq(from = tmp.vector1[tmp.check[1]], to = tmp.vector1[tmp.check[2]], length.out = (tmp.check[2] - tmp.check[1]) + 1)
        #     # Getting only the 2nd to nth values
        #     tmp.int <- tmp.int[2:(length(tmp.int)-1)]
        #     # Now weighting missing ingredients by the values they would be if interpolation was correct
        #     tmp.vector1[(tmp.check[1] + 1) : (tmp.check[2] -1)] <- # Index of missing values
        #       (100-tmp.sum) * # Total missing percent
        #       tmp.int / # This is what the interpolation would be if the total ingredient list didn't sum to 100
        #       sum(tmp.int) # Then dividing by sum of this interpolation to get percent of missing ingredients
        #   }
        # }
        
        
        ###
        # Updating percentages of last n ingredients after the last ingredient that has a percent recognized
        if(max(tmp.check) < nrow(tmp.dat)) {
          
          tmp.keep[length(tmp.keep)] <- 'YES'
          # Getting how these would interpolate, assuming the 10th ingredient is .1%
          tmp.int <- seq(from = tmp.vector1[tmp.check[length(tmp.check)]], # Value of the last recognized ingredient
                         to = min(c(.1,tmp.vector1[tmp.check[length(tmp.check)]])), # Assuming that the last ingredient is the lesser of 1% or last identified % composition
                         length.out = (nrow(tmp.dat)-tmp.check[length(tmp.check)] + 1)) # Number of ingredients needed
          
          if(length(tmp.int) >= 1) { # Logic check in case there aren't any more ingredients to identify
            # Updating tmp.vector
            tmp.vector1[(tmp.check[length(tmp.check)] + 1) : (tmp.check[length(tmp.check)] + length(tmp.int) - 1)] <-
              tmp.int[2:length(tmp.int)]
          }
        }
        
        
        
        ###
        # Updating percentages or first n ingredients if only one percentage is recognized
        if(min(tmp.check) != 1) { # Logic statement to identify products where the first identified ingredient is not the first
          tmp.sum <- sum(tmp.vector1, na.rm = TRUE) # total composition of ingredients
          tmp.int <- seq(from = (100-tmp.sum), # Percent not yet recognized
                         to = tmp.vector1[min(tmp.check)], # Percent composition of the first product
                         length.out = (min(tmp.check)))
          if(length(tmp.int)>=1) {
            tmp.vector1[1:(min(tmp.check))] <- tmp.int
          }
        }
        
        ###
        # Looping through to correct values where i.e. v1 < v2, etc
        # But only adjusting the non-keep percentages
        for(z in (length(tmp.vector1) - 1):1) {
          if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
            tmp.vector1[z] <- tmp.vector1[z+1]
          }
        }
        
        ###
        # Assuming the last ingredient is correct
        # Need to do this to provide a counterpoint for everything else
        # And also serves as a basis to solve the rest of this puzzle
        
        # First need to check to see if the last ingredient is less than all other ingredients
        if(!is.na(tmp.vector1[length(tmp.vector1)]) & # if the last ingredient isn't the smallest estimated composition
           tmp.vector1[length(tmp.vector1)] != min(tmp.vector1, na.rm = TRUE)) { # then updating this
          tmp.vector1[length(tmp.vector1)] <- min(tmp.vector1, na.rm = TRUE)
        }
        
        # And a few other checks before setting the last ingredient to "YES"
        # This is needed to benchmark the algorithm in case no or limited ingredients are known
        if(sum(tmp.vector1, na.rm = TRUE) > 100) {
          # Dividing unkowns by remaining left needed to sum to 100
          tmp.remaining.check <- 100 - sum(tmp.vector1[tmp.keep %in% 'YES'], na.rm = TRUE)
          # if this results in a negative number, setting the composition of the last ingredient to .1
          if(tmp.remaining.check %in% 0) {
            tmp.vector1[length(tmp.vector1)] <- 0.01
          } else if(tmp.remaining.check > 0) { # This scales the last ingredient proportionately based on what is left for all ingredients and the estimated composition of other ingredients that aren't known
            tmp.vector1[length(tmp.vector1)] <-
              tmp.vector1[length(tmp.vector1)] *
              (tmp.remaining.check / sum(tmp.vector1[tmp.keep %in% 'NO']))
          } else { # Else setting it to .1
            tmp.vector1[length(tmp.vector1)] <- .1
          }
        }
        
        # And now setting it to yes
        tmp.keep[length(tmp.keep)] <- 'YES'
        ###
        # And now correcting percent composition when they do not sum to 100
        # Doing this in a series of while loops
        # Where logic is
        # (a) Adjust the "non-keep" ingredients such that everything sums to 100
        # (b) check that e.g. v1 >= v2, etc
        # (c) check that e.g. v2 >= 100/2
        counter = 1
        while(sum(tmp.vector1) != 100 & counter <= 10) {
          counter = counter + 1
          if(round(sum(tmp.vector1)) != 100) {
            # Getting sum percent of the interpolated ingredients
            tmp.sum.int <- sum(tmp.vector1[tmp.keep == 'NO'])
            tmp.sum <- sum(tmp.vector1[tmp.keep == 'YES'])
            # And scaling so that all ingredients sum to 100
            tmp.vector1[tmp.keep == 'NO'] <-
              tmp.vector1[tmp.keep == 'NO'] *
              ((100 - tmp.sum) / tmp.sum.int)
            
            tmp.vector1[is.nan(tmp.vector1)] <- 0
          } # End if statement
          
          ###
          # Doing two few checks, specifically 
          # (1) %1 > %2 > %3 etc
          # Looping through to correct values where i.e. v1 < v2, etc
          # But only adjusting the non-keep percentages
          for(z in (length(tmp.vector1) - 1):1) {
            if(tmp.vector1[z] < tmp.vector1[z+1] & tmp.keep[z] != 'YES') {
              tmp.vector1[z] <- tmp.vector1[z+1]
            }
          }
          
          
          # (2)
          # Updating keep percent when possible
          for(z in (length(tmp.vector1) - 1) : 1) {
            if(tmp.vector1[z] == tmp.vector1[z+1] &
               tmp.keep[z+1] == 'YES') {
              tmp.keep[z] <- 'YES'
            }
          }
          
          # (3) %n msut be less than 100/n
          # But only adjusting the non-identified ingredients
          # And then flagging ones that cannot change again
          for(i in length(tmp.vector1):2) {
            if(tmp.vector1[i] > 100/i & tmp.keep[i] == 'NO') {
              tmp.vector1[i] <- 100/i
            } # End for loop for ingredients
          }
          
        } # End while loop
      } # End of big if else loop
      
      
      
      # updating values in the data frame
      dat$percent[dat$id.new %in% j] <-
        tmp.vector1
      dat$tot_percent[dat$id.new %in% j] <- sum(tmp.vector1, na.rm = TRUE)
    } # End of loop for products
    
    # Dropping id.new
    # dat <- dat %>% dplyr::select(-id.new)
    # Returning data frame
    return(dat)
  } # End function for ingredient interpolating
